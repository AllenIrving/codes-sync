import tensorflow as tf
import numpy as np
from PIL import Image
from glob import glob
import os
import sys

def get_img_list(dir):
	img_list=glob(os.path.join(dir,'*.png'))
	return img_list

def get_tfrecord_list(dir):
	tfrecord_list=[]
	file_list=os.listdir(dir)
	for i in range(len(file_list)):
		current_file_abs_path=os.path.abspath(file_list[i])
		if current_file_abs_path.endswith(".tfrecords"):
			tfrecord_list.append(current_file_abs_path)
		else:
			pass
	return tfrecord_list

def _int64_feature(value):
	return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
	return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def encode_to_tfrecords(root_dir,out_name='training',crop_size=33,crop_stride=14,scale=3):
	num_example=0
	dir_name=os.path.join(root_dir,out_name)
	if not os.path.exists(dir_name):
		os.mkdir(dir_name)

	input_dir=root_dir+'/input'
	label_dir=root_dir+'/label'
	input_list=get_img_list(input_dir)
	label_list=get_img_list(label_dir)
	list_len=len(input_list)

	for i in range(list_len):
		print("processing #%d/%d..." % (i,list_len))
		ori_input=Image.open(os.path.join(input_dir,input_list[i]))
		ori_input=np.array(ori_input)
		ori_label=Image.open(os.path.join(label_dir,label_list[i]))
		ori_label=np.array(ori_label)
		hei,wid,c=ori_input.shape
		hei=hei-hei%scale
		wid=wid-wid%scale
		for x in range(0,hei-crop_size,crop_stride):
			for y in range(0,wid-crop_size,crop_stride):
				if not num_example%10000:
					if num_example>0:
						writer.close()
						sys.stdout.flush()
					tfrecords_file_name=dir_name+'/'+out_name+'_patch'+str(num_example//10000)+'.tfrecords'
					writer=tf.python_io.TFRecordWriter(tfrecords_file_name)
				patch_in=ori_input[x:x+crop_size,y:y+crop_size,:]
				patch_label=ori_label[x:x+crop_size,y:y+crop_size,:]
				example=tf.train.Example(features=tf.train.Features(feature={
					'input':_bytes_feature(patch_in.tobytes()),
					'label':_bytes_feature(patch_label.tobytes()),
					'size':_int64_feature(crop_size)
					}))
				serialized=example.SerializeToString()
				writer.write(serialized)
				num_example+=1

	print("number of example:",num_example)
	writer.close()
	sys.stdout.flush()

def read_and_decode(filename_queue):
	reader=tf.TFRecordReader()
	_,serialized_example=reader.read(filename_queue)
	features=tf.parse_single_example(
		serialized_example,
		features={
		'input':tf.FixedLenFeature([],tf.string),
		'label':tf.FixedLenFeature([],tf.string),
		'size':tf.FixedLenFeature([],tf.int64)
		})
	image=tf.decode_raw(features['input'],tf.uint8)
	label=tf.decode_raw(features['label'],tf.uint8)
	size=tf.cast(features['size'],tf.int32)
	image_shape=tf.pack([size,size,3])
	image=tf.reshape(image,image_shape)
	label=tf.reshape(label,image_shape)
	images,labels=tf.train.shuffle_batch([image,label],
					batch_size=2,
					capacity=30,
					num_threads=2,
					min_after_dequeue=10)
	return images,labels	

if __name__=="__main__":
	tfList=get_tfrecord_list("/home/allen/datasets/train/training")
	for i in range(len(tfList)):
		print(tfList[i])
