import tensorflow as tf
import numpy as np
from glob import glob
import skimage.io as io
import os
import sys
import operator

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
		name=input_list[i].split('/')[-1]
		print("processing #%d/%d, filename: %s" % (i,list_len,name))
		
		ori_input=io.imread(input_list[i])
		ori_input=np.array(ori_input)
		ori_label=io.imread(label_list[i])
		ori_label=np.array(ori_label)
		hei,wid,c=ori_input.shape
		print(ori_input.shape)
		hei=hei-hei%scale
		wid=wid-wid%scale
		for x in range(0,hei-crop_size+1,crop_stride):
			for y in range(0,wid-crop_size+1,crop_stride):
				if not num_example%10000:
					if num_example>0:
						writer.close()
						sys.stdout.flush()
					tfrecords_file_name=dir_name+'/'+out_name+'_patch'+str(num_example//10000)+'.tfrecords'
					writer=tf.python_io.TFRecordWriter(tfrecords_file_name)
				patch_in=ori_input[x:x+crop_size,y:y+crop_size,0:3]
				patch_label=ori_label[x:x+crop_size,y:y+crop_size,0:3]
				#print(patch_in.shape,patch_label.shape)
				example=tf.train.Example(features=tf.train.Features(feature={
					'input':_bytes_feature(patch_in.tostring()),
					'label':_bytes_feature(patch_label.tostring()),
					'size':_int64_feature(crop_size)
					}))
				serialized=example.SerializeToString()
				writer.write(serialized)
				num_example+=1

	print("number of example:",num_example)
	writer.close()
	sys.stdout.flush()
	
def verify_tfrecord_file(filename):
	reconstructed_imgs=[]
	record_iterator=tf.python_io.tf_record_iterator(path=filename)
	for string_record in record_iterator:
		example=tf.train.Example()
		example.ParseFromString(string_record)
		img_string=(example.features.feature['input']
					.bytes_list
					.value[0])
		label_string=(example.features.feature['label']
					.bytes_list
					.value[0])
		size=int(example.features.feature['size']
					.int64_list
					.value[0])
		img_1d=np.fromstring(img_string,dtype=np.int8)
		label_1d=np.fromstring(label_string,dtype=np.int8)
		print(img_1d.shape,label_1d.shape)
		reconstructed_img=img_1d.reshape((size,size,-1))
		reconstructed_label=label_1d.reshape((size,size,-1))
		reconstructed_imgs.append((reconstructed_img,reconstructed_label))
	return reconstructed_imgs

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
	image_shape=tf.stack([size,size,3])
	image=tf.reshape(image,image_shape)
	label=tf.reshape(label,image_shape)
	image=tf.image.resize_image_with_crop_or_pad(image=image,target_height=33,target_width=33)
	label=tf.image.resize_image_with_crop_or_pad(image=label,target_height=33,target_width=33)
	
	images,labels=tf.train.shuffle_batch([image,label],
					batch_size=2,
					capacity=30,
					num_threads=2,
					min_after_dequeue=10)
	return images,labels	

if __name__=="__main__":
	#tfList=get_tfrecord_list("/home/allen/datasets/train/training")
	tf_name="/home/allen/datasets/train/training/training_patch0.tfrecords"
	#reconstructed_imgs=verify_tfrecord_file(tf_name)
	
	
	filename_queue=tf.train.string_input_producer([tf_name],num_epochs=10)
	image,label=read_and_decode(filename_queue)

	with tf.Session() as sess:
		sess.run(tf.local_variables_initializer())
		sess.run(tf.global_variables_initializer())
		
		coord=tf.train.Coordinator()
		threads=tf.train.start_queue_runners(coord=coord)
		for i in range(3):
			img,lab=sess.run([image,label])
			print(img[0,:,:,:].shape)
			print('current batch')
			io.imshow(img[0,:,:,:])
			io.show()
			io.imshow(lab[0,:,:,:])
			io.show()
			
		coord.request_stop()
		coord.join(threads)
	
