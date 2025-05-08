// Helper function for loading MNIST data I got from https://drive.google.com/file/d/11ZiNnV3YtpZ7d9afHZg0rtDRrmhha-1E/view
// 100% written by ChatGPT

use nalgebra::DMatrix;
use std::fs::File;
use std::io::{Read, Cursor, Error, ErrorKind};
use flate2::read::GzDecoder; 
use byteorder::{BigEndian, ReadBytesExt}; 

const IMAGE_MAGIC_NUMBER: u32 = 2051; // MNIST image signature
const LABEL_MAGIC_NUMBER: u32 = 2049; // MNIST label signature
const IMAGE_WIDTH: usize = 28;
const IMAGE_HEIGHT: usize = 28;
const NUM_CLASSES: usize = 10;

fn read_u32_be(reader: &mut impl Read) -> Result<u32, Error> {
    reader.read_u32::<BigEndian>()
}

pub fn load_mnist_images(path: &str) -> Result<DMatrix<f32>, Error> {
    let mut file = File::open(path)?;
    let mut raw_contents = Vec::new();
    file.read_to_end(&mut raw_contents)?;
    
    let contents = if path.ends_with(".gz") { // Simplified .gz check
        let mut decoder = GzDecoder::new(Cursor::new(raw_contents));
        let mut decompressed_contents = Vec::new();
        decoder.read_to_end(&mut decompressed_contents)?;
        decompressed_contents
    } else {
        raw_contents
    };
    
    let mut cursor = Cursor::new(contents);

    let magic_number = read_u32_be(&mut cursor)?;
    if magic_number != IMAGE_MAGIC_NUMBER {
        return Err(Error::new(ErrorKind::InvalidData, format!("Invalid magic number for images file {}: expected {}, got {}", path, IMAGE_MAGIC_NUMBER, magic_number)));
    }

    let num_images = read_u32_be(&mut cursor)? as usize;
    let num_rows = read_u32_be(&mut cursor)? as usize;
    let num_cols = read_u32_be(&mut cursor)? as usize;

    if num_rows != IMAGE_HEIGHT || num_cols != IMAGE_WIDTH {
        return Err(Error::new(ErrorKind::InvalidData, "Image dimensions are not 28x28"));
    }

    let image_size = IMAGE_WIDTH * IMAGE_HEIGHT;
    let mut image_data = Vec::with_capacity(num_images * image_size);

    for _ in 0..num_images {
        for _ in 0..image_size {
            let pixel = cursor.read_u8()?;
            image_data.push(pixel as f32 / 255.0); 
        }
    }
    
    Ok(DMatrix::from_row_slice(num_images, image_size, &image_data))
}

pub fn load_mnist_labels(path: &str, one_hot: bool) -> Result<DMatrix<f32>, Error> {
    let mut file = File::open(path)?;
    let mut raw_contents = Vec::new();
    file.read_to_end(&mut raw_contents)?;

    let contents = if path.ends_with(".gz") { // Simplified .gz check
        let mut decoder = GzDecoder::new(Cursor::new(raw_contents));
        let mut decompressed_contents = Vec::new();
        decoder.read_to_end(&mut decompressed_contents)?;
        decompressed_contents
    } else {
        raw_contents
    };

    let mut cursor = Cursor::new(contents);

    let magic_number = read_u32_be(&mut cursor)?;
    if magic_number != LABEL_MAGIC_NUMBER {
        return Err(Error::new(ErrorKind::InvalidData, format!("Invalid magic number for labels file {}: expected {}, got {}", path, LABEL_MAGIC_NUMBER, magic_number)));
    }

    let num_labels = read_u32_be(&mut cursor)? as usize;
    let mut label_data = Vec::new();

    if one_hot {
        label_data.reserve(num_labels * NUM_CLASSES);
        for _ in 0..num_labels {
            let label_val = cursor.read_u8()?;
            if label_val >= NUM_CLASSES as u8 {
                return Err(Error::new(ErrorKind::InvalidData, format!("Label {} out of bounds for {} classes", label_val, NUM_CLASSES)));
            }
            let mut one_hot_vec = vec![0.0; NUM_CLASSES];
            one_hot_vec[label_val as usize] = 1.0;
            label_data.extend_from_slice(&one_hot_vec);
        }
        Ok(DMatrix::from_row_slice(num_labels, NUM_CLASSES, &label_data))
    } else {
        label_data.reserve(num_labels);
        for _ in 0..num_labels {
            let label_val = cursor.read_u8()?;
            if label_val >= NUM_CLASSES as u8 {
                 return Err(Error::new(ErrorKind::InvalidData, format!("Label {} out of bounds for {} classes", label_val, NUM_CLASSES)));
            }
            label_data.push(label_val as f32);
        }
        Ok(DMatrix::from_column_slice(num_labels, 1, &label_data))
    }
}

// Helper to get a mini-batch
pub fn get_mini_batch(
    data: &DMatrix<f32>,
    targets: &DMatrix<f32>,
    indices: &[usize],
) -> (DMatrix<f32>, DMatrix<f32>) {
    let batch_size = indices.len();
    if batch_size == 0 {
        return (DMatrix::zeros(0,data.ncols()), DMatrix::zeros(0,targets.ncols()));
    }
    let num_features = data.ncols();
    let num_target_cols = targets.ncols();

    let mut batch_data_vec = Vec::with_capacity(batch_size * num_features);
    let mut batch_targets_vec = Vec::with_capacity(batch_size * num_target_cols);

    for &idx in indices {
        // data.row(idx) returns a RowVectorSlice. Iterate its elements.
        // The elements are references, so clone/copy them.
        batch_data_vec.extend(data.row(idx).iter().copied());
        batch_targets_vec.extend(targets.row(idx).iter().copied());
    }
    
    let batch_data = DMatrix::from_row_slice(batch_size, num_features, &batch_data_vec);
    let batch_targets = DMatrix::from_row_slice(batch_size, num_target_cols, &batch_targets_vec);
    
    (batch_data, batch_targets)
}