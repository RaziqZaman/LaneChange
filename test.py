#!/usr/bin/env python3
"""Read and print records from a TFRecord file.

Tries to use TensorFlow's TFRecordDataset to print parsed
tf.train.Example messages. If TensorFlow isn't installed,
falls back to a simple raw-reader that prints record lengths
and a hex/text preview of each record.

Usage: python test.py [path/to/file.tfrecord] [--max N]
"""

from __future__ import annotations

import os
import sys
import json
import base64


def read_with_tensorflow(path: str, max_records: int) -> None:
	import tensorflow as tf

	def feature_value_to_py(fv: 'tf.train.Feature'):
		# Determine which list is populated and convert
		if fv.bytes_list.value:
			out = []
			for b in fv.bytes_list.value:
				try:
					s = b.decode('utf-8')
					out.append(s)
				except Exception:
					out.append({'__base64__': base64.b64encode(b).decode('ascii')})
			return out
		if fv.float_list.value:
			return list(fv.float_list.value)
		if fv.int64_list.value:
			return list(fv.int64_list.value)
		return None

	def example_to_dict(ex: 'tf.train.Example') -> dict:
		result: dict = {}
		features = ex.features.feature
		for key in features:
			result[key] = feature_value_to_py(features[key])
		return result

	ds = tf.data.TFRecordDataset(path)
	printed = 0
	for i, raw in enumerate(ds):
		b = raw.numpy()
		ex = tf.train.Example()
		try:
			ex.ParseFromString(b)
			obj = example_to_dict(ex)
			print(f"--- Record {i} (parsed tf.train.Example) ---")
			if obj:
				print(json.dumps(obj, indent=2, ensure_ascii=False))
			else:
				# If the example_to_dict produced an empty dict, print the
				# raw protobuf text representation so we can inspect field
				# names and types (helps identify non-standard messages).
				print("(no top-level features found in JSON conversion)")
				print("Raw Example protobuf:\n", str(ex))
		except Exception:
			print(f"--- Record {i} (raw bytes, could not parse as Example) ---")
			print(repr(b[:200]))

		printed += 1
		if printed >= max_records:
			break


def read_raw_tfrecord(path: str, max_records: int) -> None:
	# Minimal TFRecord reader that ignores CRC checks and prints a preview.
	# TFRecord format: uint64 length (little-endian), uint32 length_crc, data, uint32 data_crc
	import struct

	def _read_uint64(f):
		b = f.read(8)
		if len(b) != 8:
			return None
		return struct.unpack('<Q', b)[0]

	printed = 0
	with open(path, 'rb') as f:
		idx = 0
		while printed < max_records:
			length = _read_uint64(f)
			if length is None:
				break
			# skip length crc
			f.read(4)
			data = f.read(length)
			# skip data crc
			f.read(4)

			print(f"--- Record {idx} ---")
			print(f"length: {length}")
			# show a short hex preview
			preview = data[:256]
			print("hex preview:", preview.hex())
			# try to decode as utf-8 (many protos won't be text, but this may help)
			try:
				text = preview.decode('utf-8')
				print("utf-8 preview:", text)
			except Exception:
				print("utf-8 preview: (binary data)")

			printed += 1
			idx += 1



def main():
	# Process all TFRecord files in the data/ folder
	data_dir = 'data'
	DEFAULT_MAX = 6

	# Get all TFRecord files in the data directory
	tfrecord_files = []
	if os.path.exists(data_dir):
		for file in os.listdir(data_dir):
			# Match common TFRecord naming patterns (e.g. *.tfrecord or *.tfrecord-000XX-of-XXXXX)
			if '.tfrecord' in file:
				tfrecord_files.append(os.path.join(data_dir, file))
	
	if not tfrecord_files:
		print(f"Error: no TFRecord files found in {data_dir}/")
		sys.exit(2)
		
	print(f"Found {len(tfrecord_files)} TFRecord files to process")

	# Prefer TensorFlow (we'll use it to parse TFExample and attempt to
	# deserialize bytes into tensors when possible).
	try:
		import tensorflow as tf  # type: ignore
	except Exception:
		print("TensorFlow is not installed or failed to import. Falling back to raw reader.")
		print("To get parsed tf.train.Example output, install TensorFlow in your venv:")
		print("  pip install tensorflow")
		print()
		for file_path in tfrecord_files:
			print(f"\nProcessing file: {file_path} with raw reader")
			read_raw_tfrecord(file_path, DEFAULT_MAX)
		return

	# Use an enhanced reader that will try to decode bytes_list entries that
	# were produced by tf.io.serialize_tensor (common in Waymo TFExample).
	MAX_SAMPLE = 24
	MAX_STR_CHARS = 360

	def try_deserialize_tensor_summary(b: bytes):
		# Return only dtype, shape and length (no values).
		for dtype in (tf.float32, tf.float64, tf.int64, tf.int32, tf.uint8):
			try:
				t = tf.io.parse_tensor(b, out_type=dtype)
				arr = t.numpy()
				return {
					'__tensor__': True,
					'dtype': str(dtype.name),
					'shape': list(arr.shape),
					'length': int(arr.size),
				}
			except Exception:
				continue
		return None

	def try_deserialize_tensor_full(b: bytes):
		# Return the full tensor values (may be large) for saving to verbose JSON.
		for dtype in (tf.float32, tf.float64, tf.int64, tf.int32, tf.uint8):
			try:
				t = tf.io.parse_tensor(b, out_type=dtype)
				arr = t.numpy()
				return {
					'__tensor__': True,
					'dtype': str(dtype.name),
					'shape': list(arr.shape),
					'values': arr.tolist(),
				}
			except Exception:
				continue
		return None

	def summarize_sequence(seq):
		length = len(seq)
		return {'__list__': True, 'length': length}

	def full_sequence(seq):
		return list(seq)

	def feature_value_to_py_waymo_summary(fv: 'tf.train.Feature'):
		# Structure-only conversion (no actual values)
		if fv.bytes_list.value:
			# If bytes represent a serialized tensor, report its shape/dtype/length.
			first = fv.bytes_list.value[0]
			des = try_deserialize_tensor_summary(first)
			if des is not None:
				return des if len(fv.bytes_list.value) == 1 else [try_deserialize_tensor_summary(b) for b in fv.bytes_list.value]
			# Otherwise report as string/base64 and length only
			return {'__bytes_list__': True, 'count': len(fv.bytes_list.value)}
		if fv.float_list.value:
			return summarize_sequence(fv.float_list.value)
		if fv.int64_list.value:
			return summarize_sequence(fv.int64_list.value)
		return None

	def feature_value_to_py_waymo_full(fv: 'tf.train.Feature'):
		# Full conversion (include values) for saving to verbose JSON
		if fv.bytes_list.value:
			out = []
			for b in fv.bytes_list.value:
				des = try_deserialize_tensor_full(b)
				if des is not None:
					out.append(des)
					continue
				try:
					s = b.decode('utf-8')
					out.append({'__string__': True, 'value': s})
				except Exception:
					out.append({'__base64__': base64.b64encode(b).decode('ascii')})
			return out if len(out) != 1 else out[0]
		if fv.float_list.value:
			return full_sequence(fv.float_list.value)
		if fv.int64_list.value:
			return full_sequence(fv.int64_list.value)
		return None

	# Small wrapper to use the Waymo-aware feature conversion
	def read_waymo(file_paths: list[str], max_records_per_file: int):
		import tensorflow as tf
		total_printed = 0
		total_verbose_records = []
		total_header_records = []
		
		for file_path in file_paths:
			print(f"\nProcessing file: {file_path}")
			ds = tf.data.TFRecordDataset(file_path)
			printed = 0
			for i, raw in enumerate(ds):
				b = raw.numpy()
				ex = tf.train.Example()
				try:
					ex.ParseFromString(b)
					# Convert features into a structure-only summary (for printing)
					features = ex.features.feature
					obj_summary = {k: feature_value_to_py_waymo_summary(features[k]) for k in features}
					# Convert features into a full verbose form (for saving)
					obj_full = {k: feature_value_to_py_waymo_full(features[k]) for k in features}
					# Add some top-level metadata if present
					if 'scenario/id' in obj_full and isinstance(obj_full['scenario/id'], dict) and '__string__' in obj_full['scenario/id']:
						sid = obj_full['scenario/id'].get('value')
					else:
						sid = None
					record_meta = {
						'record_index': total_printed,
						'file_path': file_path,
						'file_record_index': i
					}
					if sid:
						record_meta['scenario_id'] = sid

					print(f"--- Record {i} (structure summary) ---")
					if obj_summary:
						# Print only the structure (no values)
						print(json.dumps(obj_summary, indent=2, ensure_ascii=False))
					else:
						print("(no top-level features found in summary conversion)")
						print("Raw Example protobuf:\n", str(ex))
					# Save verbose record and header (structure-only)
					total_verbose_records.append({'meta': record_meta, 'features': obj_full})
					total_header_records.append({'meta': record_meta, 'features': obj_summary})
				except Exception:
					print(f"--- Record {i} (raw bytes, could not parse as Example) ---")
					print(repr(b[:200]))

				printed += 1
				total_printed += 1
				if printed >= max_records_per_file:
					break
					
			print(f"Processed {printed} records from {file_path}")

		# Write full verbose parsed records to test.json and structure-only headers to test_headers.json
		try:
			out_path = 'test.json'
			with open(out_path, 'w', encoding='utf-8') as fo:
				json.dump(total_verbose_records, fo, indent=2, ensure_ascii=False)
			print(f"\nWrote {len(total_verbose_records)} verbose parsed records to {out_path}")
		except Exception as we:
			print(f"Failed to write verbose JSON: {we}")
		try:
			hdr_path = 'test_headers.json'
			with open(hdr_path, 'w', encoding='utf-8') as fo:
				json.dump(total_header_records, fo, indent=2, ensure_ascii=False)
			print(f"Wrote {len(total_header_records)} structure-only headers to {hdr_path}")
		except Exception as he:
			print(f"Failed to write headers JSON: {he}")

	try:
		read_waymo(tfrecord_files, DEFAULT_MAX)
	except Exception as e:
		print(f"Failed reading with TensorFlow: {e}\nFalling back to raw reader.")
		for file_path in tfrecord_files:
			print(f"\nProcessing file: {file_path} with raw reader")
			read_raw_tfrecord(file_path, DEFAULT_MAX)
if __name__ == '__main__':
	main()

