/// CRF1d binary model writer. Produces byte-identical files to the C implementation.

use crate::cqdb::CqdbWriter;
use crate::crf1d::feature::{Feature, FeatureRefs, FT_STATE};

const HEADER_SIZE: usize = 48;
const CHUNK_HEADER_SIZE: usize = 12;

fn write_u32(buf: &mut Vec<u8>, val: u32) {
    buf.extend_from_slice(&val.to_le_bytes());
}

fn write_f64(buf: &mut Vec<u8>, val: f64) {
    buf.extend_from_slice(&val.to_le_bytes());
}

fn align4(buf: &mut Vec<u8>) {
    while buf.len() % 4 != 0 {
        buf.push(0);
    }
}

/// Write a complete CRF1d model file.
///
/// Parameters:
/// - `features`: all generated features
/// - `weights`: weight for each feature (same length as features)
/// - `label_strings`: label ID → string
/// - `attr_strings`: attribute ID → string
/// - `label_refs`: for each label, list of transition feature IDs
/// - `attr_refs`: for each attribute, list of state feature IDs
pub fn write_model(
    features: &[Feature],
    weights: &[f64],
    label_strings: &[String],
    attr_strings: &[String],
    label_refs: &[FeatureRefs],
    attr_refs: &[FeatureRefs],
) -> Vec<u8> {
    let num_labels = label_strings.len();
    let num_attrs_orig = attr_strings.len();
    let num_features_orig = features.len();

    // Build feature map and attribute map (prune zero-weight features)
    let mut fmap: Vec<i32> = vec![-1; num_features_orig];
    let mut amap: Vec<i32> = vec![-1; num_attrs_orig];
    let mut j = 0i32; // active feature count
    let mut b = 0i32; // active attribute count

    for (k, w) in weights.iter().enumerate() {
        if *w != 0.0 {
            fmap[k] = j;
            j += 1;
            if features[k].ftype == FT_STATE as i32 {
                let src = features[k].src as usize;
                if src < num_attrs_orig && amap[src] < 0 {
                    amap[src] = b;
                    b += 1;
                }
            }
        }
    }

    let _active_features = j as usize;
    let active_attrs = b as usize;

    let mut buf: Vec<u8> = Vec::new();

    // Reserve header space
    buf.resize(HEADER_SIZE, 0);

    // ── Features chunk ─────────��────────────────────────────────────────
    let off_features = buf.len() as u32;

    // Reserve chunk header (12 bytes), will backfill
    let feat_chunk_start = buf.len();
    buf.resize(feat_chunk_start + CHUNK_HEADER_SIZE, 0);

    let mut feat_count = 0u32;
    for (k, w) in weights.iter().enumerate() {
        if *w != 0.0 {
            let f = &features[k];
            let src = if f.ftype == FT_STATE as i32 {
                amap[f.src as usize]
            } else {
                f.src as i32
            };
            write_u32(&mut buf, f.ftype as u32);
            write_u32(&mut buf, src as u32);
            write_u32(&mut buf, f.dst as u32);
            write_f64(&mut buf, *w);
            feat_count += 1;
        }
    }

    // Backfill feature chunk header
    let feat_chunk_size = (buf.len() - feat_chunk_start) as u32;
    buf[feat_chunk_start..feat_chunk_start + 4].copy_from_slice(b"FEAT");
    buf[feat_chunk_start + 4..feat_chunk_start + 8].copy_from_slice(&feat_chunk_size.to_le_bytes());
    buf[feat_chunk_start + 8..feat_chunk_start + 12].copy_from_slice(&feat_count.to_le_bytes());

    // ── Labels CQDB ─────────────────────────────────────────────────────
    let off_labels = buf.len() as u32;
    {
        let mut cqdb = CqdbWriter::new(0);
        for (lid, s) in label_strings.iter().enumerate() {
            cqdb.put(s, lid as i32);
        }
        buf.extend_from_slice(&cqdb.close());
    }

    // ── Attrs CQDB (only active) ─────────────────────────���──────────────
    let off_attrs = buf.len() as u32;
    {
        let mut cqdb = CqdbWriter::new(0);
        for (a, s) in attr_strings.iter().enumerate() {
            if amap[a] >= 0 {
                cqdb.put(s, amap[a]);
            }
        }
        buf.extend_from_slice(&cqdb.close());
    }

    // ── Label refs (LFRF) ───────────────────���───────────────────────────
    align4(&mut buf);
    let off_labelrefs = buf.len() as u32;
    write_featureref_chunk(&mut buf, b"LFRF", label_refs, &fmap, num_labels);

    // ── Attr refs (AFRF) ────────────────────────────────────────────────
    align4(&mut buf);
    let off_attrrefs = buf.len() as u32;
    write_attrref_chunk(&mut buf, b"AFRF", attr_refs, &fmap, &amap, num_attrs_orig);

    // ── Write header ───────────────────────���────────────────────────────
    let total_size = buf.len() as u32;
    buf[0..4].copy_from_slice(b"lCRF");
    buf[4..8].copy_from_slice(&total_size.to_le_bytes());
    buf[8..12].copy_from_slice(b"FOMC");
    buf[12..16].copy_from_slice(&100u32.to_le_bytes()); // version
    buf[16..20].copy_from_slice(&0u32.to_le_bytes());   // num_features (always 0 in header)
    buf[20..24].copy_from_slice(&(num_labels as u32).to_le_bytes());
    buf[24..28].copy_from_slice(&(active_attrs as u32).to_le_bytes());
    buf[28..32].copy_from_slice(&off_features.to_le_bytes());
    buf[32..36].copy_from_slice(&off_labels.to_le_bytes());
    buf[36..40].copy_from_slice(&off_attrs.to_le_bytes());
    buf[40..44].copy_from_slice(&off_labelrefs.to_le_bytes());
    buf[44..48].copy_from_slice(&off_attrrefs.to_le_bytes());

    buf
}

/// Write a feature reference chunk (LFRF or AFRF for label refs).
fn write_featureref_chunk(
    buf: &mut Vec<u8>,
    chunk_id: &[u8; 4],
    refs: &[FeatureRefs],
    fmap: &[i32],
    num_entries: usize,
) {
    let chunk_start = buf.len();
    // The C code opens with num_labels+2 offset slots for LFRF
    let offset_count = if chunk_id == b"LFRF" { num_entries + 2 } else { num_entries };

    // Reserve chunk header (12 bytes) + offset array
    let header_and_offsets_size = CHUNK_HEADER_SIZE + 4 * offset_count;
    buf.resize(chunk_start + header_and_offsets_size, 0);

    // Write data for each ref, recording offsets
    let mut offsets: Vec<u32> = Vec::with_capacity(num_entries);
    for i in 0..num_entries {
        let offset = buf.len() as u32;
        offsets.push(offset);

        // Count active features
        let active: Vec<i32> = if i < refs.len() {
            refs[i].fids.iter()
                .filter_map(|&fid| {
                    let mapped = fmap[fid as usize];
                    if mapped >= 0 { Some(mapped) } else { None }
                })
                .collect()
        } else {
            Vec::new()
        };

        write_u32(buf, active.len() as u32);
        for fid in &active {
            write_u32(buf, *fid as u32);
        }
    }

    // Backfill chunk header
    let chunk_size = (buf.len() - chunk_start) as u32;
    buf[chunk_start..chunk_start + 4].copy_from_slice(chunk_id);
    buf[chunk_start + 4..chunk_start + 8].copy_from_slice(&chunk_size.to_le_bytes());
    buf[chunk_start + 8..chunk_start + 12].copy_from_slice(&(offset_count as u32).to_le_bytes());

    // Backfill offset array
    for (i, &off) in offsets.iter().enumerate() {
        let pos = chunk_start + CHUNK_HEADER_SIZE + i * 4;
        buf[pos..pos + 4].copy_from_slice(&off.to_le_bytes());
    }
}

/// Write an attribute feature reference chunk (AFRF), only for active attributes.
fn write_attrref_chunk(
    buf: &mut Vec<u8>,
    chunk_id: &[u8; 4],
    refs: &[FeatureRefs],
    fmap: &[i32],
    amap: &[i32],
    num_attrs_orig: usize,
) {
    // Count active attributes
    let active_attrs: usize = amap.iter().filter(|&&m| m >= 0).count();
    let chunk_start = buf.len();
    let header_and_offsets_size = CHUNK_HEADER_SIZE + 4 * active_attrs;
    buf.resize(chunk_start + header_and_offsets_size, 0);

    let mut offsets: Vec<(usize, u32)> = Vec::new(); // (mapped_aid, offset)

    for a in 0..num_attrs_orig {
        if amap[a] >= 0 {
            let offset = buf.len() as u32;
            offsets.push((amap[a] as usize, offset));

            let active: Vec<i32> = if a < refs.len() {
                refs[a].fids.iter()
                    .filter_map(|&fid| {
                        let mapped = fmap[fid as usize];
                        if mapped >= 0 { Some(mapped) } else { None }
                    })
                    .collect()
            } else {
                Vec::new()
            };

            write_u32(buf, active.len() as u32);
            for fid in &active {
                write_u32(buf, *fid as u32);
            }
        }
    }

    // Backfill chunk header
    let chunk_size = (buf.len() - chunk_start) as u32;
    buf[chunk_start..chunk_start + 4].copy_from_slice(chunk_id);
    buf[chunk_start + 4..chunk_start + 8].copy_from_slice(&chunk_size.to_le_bytes());
    buf[chunk_start + 8..chunk_start + 12].copy_from_slice(&(active_attrs as u32).to_le_bytes());

    // Backfill offset array
    for (mapped_aid, off) in &offsets {
        let pos = chunk_start + CHUNK_HEADER_SIZE + mapped_aid * 4;
        buf[pos..pos + 4].copy_from_slice(&off.to_le_bytes());
    }
}
