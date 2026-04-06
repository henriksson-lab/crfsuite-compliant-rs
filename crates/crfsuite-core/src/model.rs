/// CRF1d binary model reader (and later, writer).
///
/// Model file layout:
///   [0..48]     Header (magic "lCRF", type "FOMC", version 100, counts, offsets)
///   [off_features..]  Features chunk: "FEAT" header (12 bytes) + features (20 bytes each)
///   [off_labels..]    Label CQDB chunk
///   [off_attrs..]     Attribute CQDB chunk
///   [off_labelrefs..] Label feature refs: "LFRF" header (12 bytes) + offset array + packed fid arrays
///   [off_attrrefs..]  Attribute feature refs: "AFRF" header (12 bytes) + offset array + packed fid arrays

use crate::cqdb::CqdbReader;

fn read_u32(buf: &[u8], off: usize) -> u32 {
    u32::from_le_bytes([buf[off], buf[off + 1], buf[off + 2], buf[off + 3]])
}

fn read_f64(buf: &[u8], off: usize) -> f64 {
    f64::from_le_bytes([
        buf[off], buf[off+1], buf[off+2], buf[off+3],
        buf[off+4], buf[off+5], buf[off+6], buf[off+7],
    ])
}

const HEADER_SIZE: usize = 48;
const CHUNK_HEADER_SIZE: usize = 12; // chunk_id(4) + size(4) + num(4)
const FEATURE_SIZE: usize = 20;      // type(4) + src(4) + dst(4) + weight(8)

#[derive(Debug, Clone, Copy)]
pub struct Feature {
    pub ftype: u32, // 0 = state, 1 = transition
    pub src: u32,
    pub dst: u32,
    pub weight: f64,
}

pub struct ModelReader<'a> {
    buffer: &'a [u8],
    num_features: u32,
    num_labels: u32,
    num_attrs: u32,
    off_features: u32,
    off_labels: u32,
    off_attrs: u32,
    off_labelrefs: u32,
    off_attrrefs: u32,
    labels: CqdbReader<'a>,
    attrs: CqdbReader<'a>,
    // Precomputed feature refs (avoid per-call Vec allocation)
    cached_labelrefs: Vec<Vec<i32>>,
    cached_attrrefs: Vec<Vec<i32>>,
    // Precomputed feature data (avoid per-call buffer reads)
    cached_features: Vec<Feature>,
}

impl<'a> ModelReader<'a> {
    pub fn open(buffer: &'a [u8]) -> Option<Self> {
        if buffer.len() < HEADER_SIZE {
            return None;
        }
        if &buffer[0..4] != b"lCRF" {
            return None;
        }
        // type should be "FOMC"
        if &buffer[8..12] != b"FOMC" {
            return None;
        }

        let _size = read_u32(buffer, 4);
        let _version = read_u32(buffer, 12);
        let num_labels = read_u32(buffer, 20);
        let num_attrs = read_u32(buffer, 24);
        let off_features = read_u32(buffer, 28);
        // num_features is in the FEAT chunk header, not the file header
        let num_features = read_u32(buffer, off_features as usize + 8);
        let off_labels = read_u32(buffer, 32);
        let off_attrs = read_u32(buffer, 36);
        let off_labelrefs = read_u32(buffer, 40);
        let off_attrrefs = read_u32(buffer, 44);

        // Open CQDBs — pass remaining buffer from offset (C does this)
        let labels = CqdbReader::open(&buffer[off_labels as usize..])?;
        let attrs = CqdbReader::open(&buffer[off_attrs as usize..])?;

        let mut reader = ModelReader {
            buffer,
            num_features,
            num_labels,
            num_attrs,
            off_features,
            off_labels,
            off_attrs,
            off_labelrefs,
            off_attrrefs,
            labels,
            attrs,
            cached_labelrefs: Vec::new(),
            cached_attrrefs: Vec::new(),
            cached_features: Vec::new(),
        };

        // Precompute feature refs
        reader.cached_labelrefs = (0..num_labels as i32)
            .map(|lid| reader.read_featureref(off_labelrefs, lid))
            .collect();
        reader.cached_attrrefs = (0..num_attrs as i32)
            .map(|aid| reader.read_featureref(off_attrrefs, aid))
            .collect();

        // Precompute all features
        reader.cached_features = (0..num_features)
            .map(|fid| {
                let offset = off_features as usize + CHUNK_HEADER_SIZE + FEATURE_SIZE * fid as usize;
                Feature {
                    ftype: read_u32(buffer, offset),
                    src: read_u32(buffer, offset + 4),
                    dst: read_u32(buffer, offset + 8),
                    weight: read_f64(buffer, offset + 12),
                }
            })
            .collect();

        Some(reader)
    }

    pub fn num_features(&self) -> u32 { self.num_features }
    pub fn num_labels(&self) -> u32 { self.num_labels }
    pub fn num_attrs(&self) -> u32 { self.num_attrs }

    pub fn to_label(&self, lid: i32) -> Option<&str> {
        self.labels.to_string(lid)
    }

    pub fn to_attr(&self, aid: i32) -> Option<&str> {
        self.attrs.to_string(aid)
    }

    pub fn to_lid(&self, s: &str) -> Option<i32> {
        self.labels.to_id(s)
    }

    pub fn to_aid(&self, s: &str) -> Option<i32> {
        self.attrs.to_id(s)
    }

    /// Get a feature by ID (fast, from precomputed cache).
    #[inline]
    pub fn get_feature(&self, fid: u32) -> Option<&Feature> {
        self.cached_features.get(fid as usize)
    }

    /// Get a feature by ID (from buffer, for compatibility).
    pub fn get_feature_from_buffer(&self, fid: u32) -> Option<Feature> {
        let offset = self.off_features as usize + CHUNK_HEADER_SIZE + FEATURE_SIZE * fid as usize;
        if offset + FEATURE_SIZE > self.buffer.len() {
            return None;
        }
        Some(Feature {
            ftype: read_u32(self.buffer, offset),
            src: read_u32(self.buffer, offset + 4),
            dst: read_u32(self.buffer, offset + 8),
            weight: read_f64(self.buffer, offset + 12),
        })
    }

    /// Get feature IDs associated with a label (transition features from this label).
    pub fn get_labelref(&self, lid: i32) -> &[i32] {
        if lid >= 0 && (lid as usize) < self.cached_labelrefs.len() {
            &self.cached_labelrefs[lid as usize]
        } else {
            &[]
        }
    }

    /// Get feature IDs associated with an attribute (state features for this attribute).
    pub fn get_attrref(&self, aid: i32) -> &[i32] {
        if aid >= 0 && (aid as usize) < self.cached_attrrefs.len() {
            &self.cached_attrrefs[aid as usize]
        } else {
            &[]
        }
    }

    fn read_featureref(&self, chunk_offset: u32, index: i32) -> Vec<i32> {
        let base = chunk_offset as usize;
        // Skip chunk header (12 bytes), then read offset at position `index`
        let offset_pos = base + CHUNK_HEADER_SIZE + 4 * index as usize;
        if offset_pos + 4 > self.buffer.len() {
            return Vec::new();
        }
        let data_offset = read_u32(self.buffer, offset_pos) as usize;
        if data_offset + 4 > self.buffer.len() {
            return Vec::new();
        }
        // At data_offset: num_features (u32), then num_features × feature_id (u32)
        let num = read_u32(self.buffer, data_offset) as usize;
        let mut fids = Vec::with_capacity(num);
        for i in 0..num {
            let off = data_offset + 4 + i * 4;
            if off + 4 > self.buffer.len() {
                break;
            }
            fids.push(read_u32(self.buffer, off) as i32);
        }
        fids
    }

    // ── Header fields for dump ──────────────────────────────────────────

    pub fn header_size(&self) -> u32 { read_u32(self.buffer, 4) }
    pub fn header_version(&self) -> u32 { read_u32(self.buffer, 12) }
    pub fn off_features(&self) -> u32 { self.off_features }
    pub fn off_labels(&self) -> u32 { self.off_labels }
    pub fn off_attrs(&self) -> u32 { self.off_attrs }
    pub fn off_labelrefs(&self) -> u32 { self.off_labelrefs }
    pub fn off_attrrefs(&self) -> u32 { self.off_attrrefs }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn model_bytes() -> Vec<u8> {
        let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent().unwrap().parent().unwrap().to_path_buf();
        std::fs::read(root.join("test_data/model_c.bin")).expect("model_c.bin not found")
    }

    #[test]
    fn test_model_reader_open() {
        let data = model_bytes();
        let model = ModelReader::open(&data).expect("failed to open model");
        assert_eq!(model.num_labels(), 4);
        assert!(model.num_attrs() > 0);
        assert!(model.num_features() > 0);
    }

    #[test]
    fn test_model_reader_labels() {
        let data = model_bytes();
        let model = ModelReader::open(&data).unwrap();
        // Check known labels
        let expected = ["B-NP", "I-NP", "B-VP", "B-PP"];
        for (i, label) in expected.iter().enumerate() {
            assert_eq!(model.to_label(i as i32), Some(*label));
            assert_eq!(model.to_lid(label), Some(i as i32));
        }
    }

    #[test]
    fn test_model_reader_features() {
        let data = model_bytes();
        let model = ModelReader::open(&data).unwrap();
        // Read first feature
        let f = model.get_feature(0).expect("feature 0");
        assert!(f.ftype <= 1);
        assert!(f.weight.is_finite());
    }

    #[test]
    fn test_model_reader_labelrefs() {
        let data = model_bytes();
        let model = ModelReader::open(&data).unwrap();
        // Each label should have some transition features
        for lid in 0..model.num_labels() as i32 {
            let refs = model.get_labelref(lid);
            // At least some labels should have features
            // (not all necessarily)
        }
    }

    #[test]
    fn test_model_reader_attrrefs() {
        let data = model_bytes();
        let model = ModelReader::open(&data).unwrap();
        for aid in 0..model.num_attrs().min(5) as i32 {
            let refs = model.get_attrref(aid);
            assert!(!refs.is_empty(), "attr {} should have features", aid);
            for &fid in refs {
                let f = model.get_feature(fid as u32).unwrap();
                assert_eq!(f.ftype, 0, "attr ref should point to state features");
                assert_eq!(f.src, aid as u32, "state feature src should be the attribute");
            }
        }
    }
}
