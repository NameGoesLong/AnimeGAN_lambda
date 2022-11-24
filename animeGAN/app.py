#@title Define functions
#@markdown Select model version and run.
import onnxruntime as ort
import time, cv2
from PIL import Image
from io import BytesIO

import numpy as np
import gc
import boto3
from botocore.exceptions import ClientError

pic_form = ['.jpeg','.jpg','.png','.JPEG','.JPG','.PNG']
device_name = ort.get_device()

providers = ['CPUExecutionProvider']

model = 'AnimeGANv2_Paprika' #@param ['AnimeGAN_Hayao','AnimeGANv2_Hayao','AnimeGANv2_Shinkai','AnimeGANv2_Paprika']
#load model
session = ort.InferenceSession(f'{model}.onnx', providers=providers)

def process_image(img, x32=True):
    h, w = img.shape[:2]
    if x32: # resize image to multiple of 32s
        def to_32s(x):
            return 256 if x < 256 else x - x%32
        img = cv2.resize(img, (to_32s(w), to_32s(h)))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)/ 127.5 - 1.0
    return img

def Convert(img, scale):
    x = session.get_inputs()[0].name
    y = session.get_outputs()[0].name
    fake_img = session.run(None, {x : img})[0]
    images = (np.squeeze(fake_img) + 1.) / 2 * 255
    images = np.clip(images, 0, 255).astype(np.uint8)
    output_image = cv2.resize(images, (scale[1],scale[0]))
    return cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)
    

in_dir = 'input'
out_dir = 'output'

import os
from glob import glob

def read_image_from_s3(bucket, key, region_name='ap-southeast-1'):
    """Load image file from s3.

    Parameters
    ----------
    bucket: string
        Bucket name
    key : string
        Path in s3

    Returns
    -------
    np array
        Image array
    """
    s3 = boto3.resource('s3', region_name='ap-southeast-1')
    bucket = s3.Bucket(bucket)
    object = bucket.Object(key)
    response = object.get()
    file_stream = response['Body']
    im = Image.open(file_stream)
    # load image
    img0 = np.array(im).astype(np.float32)
    # convert and crop in size of 32
    img = process_image(img0)

    img = np.expand_dims(img, axis=0)

    return img, img0.shape[:2]


def write_image_to_s3(cv2_img, bucket, key, region_name='ap-southeast-1'):
    """Write an image array into S3 bucket

    Parameters
    ----------
    bucket: string
        Bucket name
    key : string
        Path in s3

    Returns
    -------
    None
    """
    s3 = boto3.resource('s3', region_name)
    bucket = s3.Bucket(bucket)
    object = bucket.Object(key)
    file_stream = BytesIO()
    img_array = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
    im = Image.fromarray(img_array)
    im.save(file_stream, format='jpeg')
    object.put(Body=file_stream.getvalue())



def upload_file(file_name, bucket, object_name=None):
    """Upload a file to an S3 bucket

    :param file_name: File to upload
    :param bucket: Bucket to upload to
    :param object_name: S3 object name. If not specified then file_name is used
    :return: True if file was uploaded, else False
    """

    # If S3 object_name was not specified, use file_name
    if object_name is None:
        object_name = os.path.basename(file_name)

    # Upload the file
    s3_client = boto3.client('s3')
    try:
        response = s3_client.upload_file(file_name, bucket, object_name)
    except ClientError as e:
        print(e)
        return False
    return True

def handler(event, context):
    input_bucket_name = event["input_bucket"]
    s3_client = boto3.client("s3")
    response = s3_client.list_objects_v2(Bucket=input_bucket_name, Prefix= "input/")

    for obj_content in response["Contents"]:
        if obj_content["Key"][-1] != "/":
            print(obj_content["Key"])
            # read image from s3
            # image = read_image_from_s3(input_bucket_name, obj_content["Key"], "us-west-2")
            print("read file from s3")
            mat, scale = read_image_from_s3(input_bucket_name, obj_content["Key"], "us-west-2")
            gc.collect()
            print("converting the image...")
            res = Convert(mat, scale)
            _, img_name = obj_content["Key"].split("/", 1)
            print("writing image to s3")
            # write image to s3
            write_image_to_s3(res,input_bucket_name,"output/animed_"+img_name, "us-west-2")
            print("finished writing")

    return {
        "message": "exection succeed"
    }

