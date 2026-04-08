pub mod lookup3;

use std::str;

use lookup3::hash_string;

const CHUNKID: &[u8; 4] = b"CQDB";
const BYTEORDER_CHECK: u32 = 0x62445371;
const NUM_TABLES: usize = 256;
const HEADER_SIZE: usize = 24;
const TABLEREF_SIZE: usize = NUM_TABLES * 8; // 256 × (offset u32 + num u32)
const OFFSET_DATA: usize = HEADER_SIZE + TABLEREF_SIZE; // 2072

fn read_u32(buf: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes([buf[offset], buf[offset + 1], buf[offset + 2], buf[offset + 3]])
}

// ── CQDB Reader ─────────────────────────────────────────────────────────────

struct Bucket {
    hash: u32,
    offset: u32,
}

struct Table {
    buckets: Vec<Bucket>,
}

/// Read-only CQDB database backed by a byte buffer.
pub struct CqdbReader<'a> {
    buffer: &'a [u8],
    tables: Vec<Table>,
    bwd: Vec<u32>, // backward links (id → record offset), empty if ONEWAY
    num: usize,
}

impl<'a> CqdbReader<'a> {
    /// Open a CQDB from a byte buffer.
    pub fn open(buffer: &'a [u8]) -> Option<Self> {
        if buffer.len() < OFFSET_DATA {
            return None;
        }
        // Verify magic
        if &buffer[0..4] != CHUNKID {
            return None;
        }
        // Verify byte order
        let byteorder = read_u32(buffer, 12);
        if byteorder != BYTEORDER_CHECK {
            return None;
        }

        let bwd_size = read_u32(buffer, 16) as usize;
        let bwd_offset = read_u32(buffer, 20) as usize;

        // Load hash tables
        let mut tables: Vec<Table> = Vec::with_capacity(NUM_TABLES);
        let mut total_records = 0usize;
        for i in 0..NUM_TABLES {
            let ref_offset = HEADER_SIZE + i * 8;
            let tbl_offset = read_u32(buffer, ref_offset) as usize;
            let tbl_num = read_u32(buffer, ref_offset + 4) as usize;

            if tbl_offset != 0 && tbl_num != 0 {
                let mut buckets = Vec::with_capacity(tbl_num);
                for j in 0..tbl_num {
                    let bo = tbl_offset + j * 8;
                    if bo + 8 > buffer.len() {
                        return None;
                    }
                    buckets.push(Bucket {
                        hash: read_u32(buffer, bo),
                        offset: read_u32(buffer, bo + 4),
                    });
                }
                // Actual record count = tbl_num / 2 (tables are 2× overallocated)
                total_records += tbl_num / 2;
                tables.push(Table { buckets });
            } else {
                tables.push(Table { buckets: Vec::new() });
            }
        }

        // Load backward links
        let bwd = if bwd_offset != 0 && bwd_size > 0 {
            let mut v = Vec::with_capacity(bwd_size);
            for i in 0..bwd_size {
                let off = bwd_offset + i * 4;
                if off + 4 > buffer.len() {
                    return None;
                }
                v.push(read_u32(buffer, off));
            }
            v
        } else {
            Vec::new()
        };

        Some(CqdbReader {
            buffer,
            tables,
            bwd,
            num: total_records,
        })
    }

    /// Look up the integer ID for a string.
    pub fn to_id(&self, s: &str) -> Option<i32> {
        let hv = hash_string(s);
        let t = (hv % NUM_TABLES as u32) as usize;
        let table = &self.tables[t];
        let n = table.buckets.len();
        if n == 0 {
            return None;
        }

        let mut k = ((hv >> 8) % n as u32) as usize;
        loop {
            let bucket = &table.buckets[k];
            if bucket.offset == 0 {
                return None; // empty slot = not found
            }
            if bucket.hash == hv {
                // Verify string match
                let off = bucket.offset as usize;
                if off + 8 > self.buffer.len() {
                    return None;
                }
                let id = read_u32(self.buffer, off) as i32;
                let ksize = read_u32(self.buffer, off + 4) as usize;
                if off + 8 + ksize > self.buffer.len() {
                    return None;
                }
                let key = &self.buffer[off + 8..off + 8 + ksize - 1]; // exclude null
                if key == s.as_bytes() {
                    return Some(id);
                }
            }
            k = (k + 1) % n;
        }
    }

    /// Look up a string by its integer ID.
    pub fn to_string(&self, id: i32) -> Option<&str> {
        if id < 0 || (id as usize) >= self.bwd.len() {
            return None;
        }
        let offset = self.bwd[id as usize] as usize;
        if offset == 0 {
            return None;
        }
        if offset + 8 > self.buffer.len() {
            return None;
        }
        let ksize = read_u32(self.buffer, offset + 4) as usize;
        if ksize == 0 || offset + 8 + ksize > self.buffer.len() {
            return None;
        }
        let key = &self.buffer[offset + 8..offset + 8 + ksize - 1]; // exclude null
        str::from_utf8(key).ok()
    }

    /// Number of records.
    pub fn num(&self) -> usize {
        self.num
    }
}

// ── CQDB Writer ─────────────────────────────────────────────────────────────

fn write_u32(buf: &mut Vec<u8>, value: u32) {
    buf.extend_from_slice(&value.to_le_bytes());
}

struct WriterBucket {
    hash: u32,
    offset: u32,
}

struct WriterTable {
    entries: Vec<WriterBucket>,
}

/// Builds a CQDB chunk in memory, producing the same byte layout as the C writer.
pub struct CqdbWriter {
    flag: u32,
    data: Vec<u8>,     // accumulates header + tablerefs + records
    tables: Vec<WriterTable>,
    bwd: Vec<u32>,     // backward links (id → offset)
    bwd_num: usize,    // highest id+1
}

impl CqdbWriter {
    pub fn new(flag: u32) -> Self {
        // Reserve space for header (24 bytes) + table refs (2048 bytes)
        let data = vec![0; OFFSET_DATA];

        CqdbWriter {
            flag,
            data,
            tables: (0..NUM_TABLES).map(|_| WriterTable { entries: Vec::new() }).collect(),
            bwd: Vec::new(),
            bwd_num: 0,
        }
    }

    /// Add a string→id mapping. Strings should be added in order.
    pub fn put(&mut self, s: &str, id: i32) {
        let hv = hash_string(s);
        let t = (hv % NUM_TABLES as u32) as usize;

        let offset = self.data.len() as u32;

        // Write record: [id(4)] [ksize(4)] [key with null(ksize)]
        let key_bytes = s.as_bytes();
        let ksize = key_bytes.len() + 1; // include null terminator
        write_u32(&mut self.data, id as u32);
        write_u32(&mut self.data, ksize as u32);
        self.data.extend_from_slice(key_bytes);
        self.data.push(0); // null terminator

        // Add to hash table
        self.tables[t].entries.push(WriterBucket { hash: hv, offset });

        // Add backward link
        if self.flag & 1 == 0 { // not CQDB_ONEWAY
            let uid = id as usize;
            if uid >= self.bwd.len() {
                self.bwd.resize(uid + 1, 0);
            }
            self.bwd[uid] = offset;
            if uid + 1 > self.bwd_num {
                self.bwd_num = uid + 1;
            }
        }
    }

    /// Finalize and return the complete CQDB chunk as bytes.
    pub fn close(mut self) -> Vec<u8> {
        // Track where each table's buckets will be written
        let mut table_offsets: Vec<u32> = vec![0; NUM_TABLES];
        let mut table_sizes: Vec<u32> = vec![0; NUM_TABLES];

        // Write hash tables
        for i in 0..NUM_TABLES {
            let num_entries = self.tables[i].entries.len();
            if num_entries == 0 {
                continue;
            }

            let n = num_entries * 2; // double allocation
            table_offsets[i] = self.data.len() as u32;
            table_sizes[i] = n as u32;

            // Build open-addressed hash table
            let mut dst: Vec<(u32, u32)> = vec![(0, 0); n]; // (hash, offset)
            for entry in &self.tables[i].entries {
                let mut k = ((entry.hash >> 8) % n as u32) as usize;
                while dst[k].1 != 0 {
                    k = (k + 1) % n;
                }
                dst[k] = (entry.hash, entry.offset);
            }

            // Write buckets
            for (hash, offset) in &dst {
                write_u32(&mut self.data, *hash);
                write_u32(&mut self.data, *offset);
            }
        }

        // Write backward array
        let bwd_offset = if self.flag & 1 == 0 && self.bwd_num > 0 {
            let off = self.data.len() as u32;
            for i in 0..self.bwd_num {
                let v = if i < self.bwd.len() { self.bwd[i] } else { 0 };
                write_u32(&mut self.data, v);
            }
            off
        } else {
            0
        };

        let total_size = self.data.len() as u32;

        // Write header at offset 0
        self.data[0..4].copy_from_slice(CHUNKID);
        self.data[4..8].copy_from_slice(&total_size.to_le_bytes());
        self.data[8..12].copy_from_slice(&self.flag.to_le_bytes());
        self.data[12..16].copy_from_slice(&BYTEORDER_CHECK.to_le_bytes());
        self.data[16..20].copy_from_slice(&(self.bwd_num as u32).to_le_bytes());
        self.data[20..24].copy_from_slice(&bwd_offset.to_le_bytes());

        // Write table references at offset 24
        for i in 0..NUM_TABLES {
            let ref_off = HEADER_SIZE + i * 8;
            self.data[ref_off..ref_off + 4].copy_from_slice(&table_offsets[i].to_le_bytes());
            self.data[ref_off + 4..ref_off + 8].copy_from_slice(&table_sizes[i].to_le_bytes());
        }

        self.data
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cqdb_open_too_small() {
        let buf = vec![0u8; 100];
        assert!(CqdbReader::open(&buf).is_none());
    }

    #[test]
    fn test_cqdb_open_wrong_magic() {
        let mut buf = vec![0u8; OFFSET_DATA];
        buf[0..4].copy_from_slice(b"XXXX");
        assert!(CqdbReader::open(&buf).is_none());
    }

    #[test]
    fn test_cqdb_writer_reader_roundtrip() {
        let entries = vec![
            (0, "B-NP"),
            (1, "I-NP"),
            (2, "B-VP"),
            (3, "B-PP"),
        ];

        let mut writer = CqdbWriter::new(0);
        for &(id, s) in &entries {
            writer.put(s, id);
        }
        let data = writer.close();

        let reader = CqdbReader::open(&data).expect("failed to open written CQDB");
        assert_eq!(reader.num(), entries.len());

        for &(id, s) in &entries {
            assert_eq!(reader.to_id(s), Some(id), "to_id({:?}) failed", s);
            assert_eq!(reader.to_string(id), Some(s), "to_string({}) failed", id);
        }
        assert!(reader.to_id("NONEXISTENT").is_none());
    }

    #[test]
    fn test_cqdb_writer_many_entries() {
        let mut writer = CqdbWriter::new(0);
        let entries: Vec<String> = (0..100).map(|i| format!("entry_{:04}", i)).collect();
        for (id, s) in entries.iter().enumerate() {
            writer.put(s, id as i32);
        }
        let data = writer.close();

        let reader = CqdbReader::open(&data).expect("failed to open CQDB with 100 entries");
        assert_eq!(reader.num(), 100);

        for (id, s) in entries.iter().enumerate() {
            assert_eq!(reader.to_id(s), Some(id as i32));
            assert_eq!(reader.to_string(id as i32), Some(s.as_str()));
        }
    }
}
