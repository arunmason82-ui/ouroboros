import fs from 'fs-extra'

import type {
  ArchitectureType,
  BaseGGUFMetadata,
  BloomMetadata,
  FalconMetadata,
  GemmaMetadata,
  GGUFMetadata,
  GPT2Metadata,
  GPTJMetadata,
  GPTNeoXMetadata,
  LlamaMetadata,
  MPTMetadata,
  RWKVMetadata,
} from './metadataTypes'

type MetadataBaseValue = string | number | bigint | boolean
type MetadataArray = MetadataBaseValue[]
type MetadataValue = MetadataBaseValue | MetadataArray

type Version = 1 | 2 | 3
const isVersion = (version: number): version is Version =>
  version === 1 || version === 2 || version === 3

type NumberBytes = { error: Error } | { error: null; value: number }
type BigIntBytes = { error: Error } | { error: null; value: bigint }

const ggufMagicNumber = Buffer.from([0x47, 0x47, 0x55, 0x46]).readInt32LE()
const fileChunkSize = 10 * 1024 * 1024

type GGUFFile = { data: Buffer; fd: number; offset: number }

/**
 * Reads GGUF file chunks incrementally to avoid allocating all data in memory at once.
 * This is critical for handling large model files (10GB+) without excessive memory usage.
 */
const readFileChunk = async (file: GGUFFile): Promise<Error | null> => {
  const buffer = Buffer.alloc(fileChunkSize)
  const { bytesRead } = await fs.read(file.fd, buffer, 0, fileChunkSize, null)
  if (bytesRead !== fileChunkSize) {
    return new Error('unexpected bytes read')
  }
  file.data = Buffer.concat([file.data, buffer])
  return null
}

const readNBytes = async (
  numBytes: number,
  file: GGUFFile,
): Promise<{ error: Error } | { bytes: Buffer; error?: undefined }> => {
  const end = file.offset + numBytes
  if (end > file.data.length) {
    const err = await readFileChunk(file)
    if (err) return { error: err }
  }
  const buffer = file.data.subarray(file.offset, end)
  file.offset = end
  return { bytes: buffer }
}

// Endian-aware buffer readers for different numeric types
const readUint8 = async (file: GGUFFile): Promise<NumberBytes> => {
  const bytes = await readNBytes(1, file)
  if (bytes.error) return bytes
  return { error: null, value: bytes.bytes.readUInt8() }
}

const readUint16 = async (file: GGUFFile): Promise<NumberBytes> => {
  const bytes = await readNBytes(2, file)
  if (bytes.error) return bytes
  return { error: null, value: bytes.bytes.readUInt16LE() }
}

const readUint32 = async (file: GGUFFile): Promise<NumberBytes> => {
  const bytes = await readNBytes(4, file)
  if (bytes.error) return bytes
  return { error: null, value: bytes.bytes.readUInt32LE() }
}

const readUint64 = async (file: GGUFFile): Promise<BigIntBytes> => {
  const bytes = await readNBytes(8, file)
  if (bytes.error) return bytes
  return { error: null, value: bytes.bytes.readBigUInt64LE() }
}

const readInt8 = async (file: GGUFFile): Promise<NumberBytes> => {
  const bytes = await readNBytes(1, file)
  if (bytes.error) return bytes
  return { error: null, value: bytes.bytes.readInt8() }
}

const readInt16 = async (file: GGUFFile): Promise<NumberBytes> => {
  const bytes = await readNBytes(2, file)
  if (bytes.error) return bytes
  return { error: null, value: bytes.bytes.readInt16LE() }
}

const readInt32 = async (file: GGUFFile): Promise<NumberBytes> => {
  const bytes = await readNBytes(4, file)
  if (bytes.error) return bytes
  return { error: null, value: bytes.bytes.readInt32LE() }
}

const readInt64 = async (file: GGUFFile): Promise<BigIntBytes> => {
  const bytes = await readNBytes(8, file)
  if (bytes.error) return bytes
  return { error: null, value: bytes.bytes.readBigInt64LE() }
}

const readFloat32 = async (file: GGUFFile): Promise<NumberBytes> => {
  const bytes = await readNBytes(4, file)
  if (bytes.error) return bytes
  return { error: null, value: bytes.bytes.readFloatLE() }
}

const readFloat64 = async (file: GGUFFile): Promise<NumberBytes> => {
  const bytes = await readNBytes(8, file)
  if (bytes.error) return bytes
  const arrayBuffer = new ArrayBuffer(8)
  const view = new DataView(arrayBuffer)
  for (let i = 0; i < 8; ++i) {
    view.setUint8(i, bytes.bytes[i])
  }
  return { error: null, value: view.getFloat64(0) }
}

const readBool = async (
  file: GGUFFile,
): Promise<{ error: Error } | { error: null; value: boolean }> => {
  const bytes = await readNBytes(1, file)
  if (bytes.error) return bytes
  return { error: null, value: !!bytes.bytes.readUint8() }
}

/**
 * Reads version-aware size values:
 * - Version 1: uint32 (32-bit)
 * - Version 2+: uint64 (64-bit) for larger model support
 */
const readVersionedSize = async (
  version: Version,
  file: GGUFFile,
): Promise<BigIntBytes> => {
  let value: bigint
  switch (version) {
    case 1: {
      const n = await readUint32(file)
      if (n.error) return n
      value = BigInt(n.value)
      break
    }
    case 3:
    case 2: {
      const n = await readUint64(file)
      if (n.error) return n
      value = n.value
      break
    }
  }
  return { error: null, value }
}

const readString = async (
  version: Version,
  file: GGUFFile,
): Promise<{ error: Error } | { error: null; value: string }> => {
  const nBytes = await readVersionedSize(version, file)
  if (nBytes.error) return nBytes
  const strBuffer = await readNBytes(Number(nBytes.value), file)
  if (strBuffer.error) return strBuffer
  return {
    error: null,
    value: strBuffer.bytes.toString().replace(/\x00/g, ''),
  }
}

/**
 * Parses GGUF metadata arrays with type discrimination.
 * Supports all GGUF data types including quantization metadata.
 */
const readArray = async (
  version: Version,
  file: GGUFFile,
): Promise<{ error: Error } | { error: null; value: MetadataArray }> => {
  const arrType = await readUint32(file)
  if (arrType.error) return arrType
  const numElts = await readVersionedSize(version, file)
  if (numElts.error) return numElts
  const ret: MetadataArray = []
  
  for (let i = 0; i < numElts.value; ++i) {
    switch (arrType.value) {
      case 0: {
        const value = await readUint8(file)
        if (value.error) return value
        ret.push(value.value)
        break
      }
      case 1: {
        const value = await readInt8(file)
        if (value.error) return value
        ret.push(value.value)
        break
      }
      case 2: {
        const value = await readUint16(file)
        if (value.error) return value
        ret.push(value.value)
        break
      }
      case 3: {
        const value = await readInt16(file)
        if (value.error) return value
        ret.push(value.value)
        break
      }
      case 4: {
        const value = await readUint32(file)
        if (value.error) return value
        ret.push(value.value)
        break
      }
      case 5: {
        const value = await readInt32(file)
        if (value.error) return value
        ret.push(value.value)
        break
      }
      case 6: {
        const value = await readFloat32(file)
        if (value.error) return value
        ret.push(value.value)
        break
      }
      case 7: {
        const value = await readBool(file)
        if (value.error) return value
        ret.push(value.value)
        break
      }
      case 8: {
        const value = await readString(version, file)
        if (value.error) return value
        ret.push(value.value)
        break
      }
      case 10: {
        const value = await readUint64(file)
        if (value.error) return value
        ret.push(value.value)
        break
      }
      case 11: {
        const value = await readInt64(file)
        if (value.error) return value
        ret.push(value.value)
        break
      }
      case 12: {
        const value = await readFloat64(file)
        if (value.error) return value
        ret.push(value.value)
        break
      }
      default: {
        return { error: new Error('unknown metadata element key type') }
      }
    }
  }

  return { error: null, value: ret }
}

/**
 * Maps GGUF quantization type codes to human-readable format strings.
 * Essential for identifying model quantization scheme (Q4_0, Q8_0, etc.).
 */
const fileTypeIntToString = (
  fileType?: number,
): string | undefined => {
  if (fileType == null) return undefined
  const typeMap: Record<number, string> = {
    0: 'ALL_F32',
    1: 'MOSTLY_F16',
    2: 'MOSTLY_Q4_0',
    3: 'MOSTLY_Q4_1',
    4: 'MOSTLY_Q4_1_SOME_F16',
    5: 'MOSTLY_Q4_2',
    6: 'MOSTLY_Q4_3',
    7: 'MOSTLY_Q8_0',
    8: 'MOSTLY_Q5_0',
    9: 'MOSTLY_Q5_1',
    10: 'MOSTLY_Q2_K',
    11: 'MOSTLY_Q3_K_S',
    12: 'MOSTLY_Q3_K_M',
    13: 'MOSTLY_Q3_K_L',
    14: 'MOSTLY_Q4_K_S',
    15: 'MOSTLY_Q4_K_M',
    16: 'MOSTLY_Q5_K_S',
    17: 'MOSTLY_Q5_K_M',
    18: 'MOSTLY_Q6_K',
    19: 'MOSTLY_IQ2_XXS',
    20: 'MOSTLY_IQ2_XS',
    21: 'MOSTLY_Q2_K_S',
    22: 'MOSTLY_Q3_K_XS',
    23: 'MOSTLY_IQ3_XXS',
  }
  return typeMap[fileType]
}

/**
 * Parses GGUF file metadata without loading full tensor data.
 * Optimized for efficient metadata-only inspection of quantized models.
 * 
 * @param filePath Path to GGUF file
 * @returns Parsed metadata with tensor information and quantization details
 */
export const parseGGUFMetadata = async (
  filePath: string,
): Promise<{ error?: Error; metadata?: Record<string, any> }> => {
  const fd = await fs.open(filePath, 'r')
  const file: GGUFFile = { data: Buffer.from([]), fd, offset: 0 }

  try {
    const magic = await readUint32(file)
    if (magic.error) return magic
    if (magic.value !== ggufMagicNumber) {
      return { error: new Error('invalid gguf magic number') }
    }

    const version = await readUint32(file)
    if (version.error) return version
    if (!isVersion(version.value)) {
      return { error: new Error(`unsupported gguf version: ${version.value}`) }
    }

    const tensorCount = await readVersionedSize(version.value, file)
    if (tensorCount.error) return tensorCount

    const numKv = await readVersionedSize(version.value, file)
    if (numKv.error) return numKv

    const metadata: Record<string, any> = {}

    const setKey = (keyName: string, value: MetadataValue) => {
      const keys = keyName.split('.')
      let obj = metadata
      for (const [index, key] of keys.entries()) {
        if (!index) continue
        const prevKey = keys[index - 1]
        if (!obj[prevKey]) obj[prevKey] = {}
        if (index === keys.length - 1) {
          if (typeof obj[prevKey] === 'object') {
            obj[prevKey][key] = value
          }
        }
        obj = obj[prevKey]
      }
    }

    for (let i = 0; i < numKv.value; ++i) {
      const key = await readString(version.value, file)
      if (key.error) return key
      const keyType = await readUint32(file)
      if (keyType.error) return keyType
      
      switch (keyType.value) {
        case 0: {
          const value = await readUint8(file)
          if (value.error) return value
          setKey(key.value, value.value)
          break
        }
        case 1: {
          const value = await readInt8(file)
          if (value.error) return value
          setKey(key.value, value.value)
          break
        }
        case 2: {
          const value = await readUint16(file)
          if (value.error) return value
          setKey(key.value, value.value)
          break
        }
        case 3: {
          const value = await readInt16(file)
          if (value.error) return value
          setKey(key.value, value.value)
          break
        }
        case 4: {
          const value = await readUint32(file)
          if (value.error) return value
          setKey(key.value, value.value)
          break
        }
        case 5: {
          const value = await readInt32(file)
          if (value.error) return value
          setKey(key.value, value.value)
          break
        }
        case 6: {
          const value = await readFloat32(file)
          if (value.error) return value
          setKey(key.value, value.value)
          break
        }
        case 7: {
          const value = await readBool(file)
          if (value.error) return value
          setKey(key.value, value.value)
          break
        }
        case 8: {
          const value = await readString(version.value, file)
          if (value.error) return value
          setKey(key.value, value.value)
          break
        }
        case 9: {
          const value = await readArray(version.value, file)
          if (value.error) return value
          setKey(key.value, value.value)
          break
        }
        case 10: {
          const value = await readUint64(file)
          if (value.error) return value
          setKey(key.value, value.value)
          break
        }
        case 11: {
          const value = await readInt64(file)
          if (value.error) return value
          setKey(key.value, value.value)
          break
        }
        case 12: {
          const value = await readFloat64(file)
          if (value.error) return value
          setKey(key.value, value.value)
          break
        }
        default: {
          return { error: new Error('unknown metadata key type') }
        }
      }
    }
    
    return { metadata }
  } finally {
    await fs.close(fd)
  }
}

export default parseGGUFMetadata
