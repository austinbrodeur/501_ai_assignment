bplist00�_WebSubresources_WebMainResource��	
^WebResourceURL_WebResourceResponse_WebResourceData_WebResourceMIMEType_)http://pages.cpsc.ucalgary.ca/favicon.icoObplist00�
X$versionY$archiverT$topX$objects ��_NSKeyedArchiver�	_WebResourceResponse��'-.4567OPQRSTUVWXYZ[\]^_`aeU$null� !"#$%&___nsurlrequest_proto_prop_obj_6___nsurlrequest_proto_prop_obj_3R$2___nsurlrequest_proto_prop_obj_0R$3___nsurlrequest_proto_prop_obj_4V$class___nsurlrequest_proto_prop_obj_1R$4R$0___nsurlrequest_proto_prop_obj_5___nsurlrequest_proto_prop_obj_2R$1���	���� ��()$+,WNS.base[NS.relative� ��_)http://pages.cpsc.ucalgary.ca/favicon.ico�/012Z$classnameX$classesUNSURL�13XNSObject#A����hU ��89:DNWNS.keysZNS.objects�;<=>?@ABC�	�
��������EGHIJKLM����������VServer\Content-TypeTEtag]Last-ModifiedX__hhaa__]Accept-RangesTDate^Content-LengthZConnection_ Apache/2.2.15 (Scientific Linux)_image/vnd.microsoft.icon_"160004-57e-4f23de4a51a94"_Wed, 12 Feb 2014 23:25:49 GMT_ 

YnBsaXN0MDDYAQIDBAUGBwgJCw0PERMVF1pDb25uZWN0aW9uVlNlcnZlclxDb250ZW50LVR5cGVdTGFzdC1Nb2RpZmllZF1BY2NlcHQtUmFuZ2VzVERhdGVeQ29udGVudC1MZW5ndGhURXRhZ6EKVWNsb3NloQxfECBBcGFjaGUvMi4yLjE1IChTY2llbnRpZmljIExpbnV4KaEOXxAYaW1hZ2Uvdm5kLm1pY3Jvc29mdC5pY29uoRBfEB1XZWQsIDEyIEZlYiAyMDE0IDIzOjI1OjQ5IEdNVKESVWJ5dGVzoRRfEB1TdW4sIDIxIEp1bCAyMDE5IDA1OjA3OjUyIEdNVKEWVDE0MDahGF8QGiIxNjAwMDQtNTdlLTRmMjNkZTRhNTFhOTQiAAgAGQAkACsAOABGAFQAWQBoAG0AbwB1AHcAmgCcALcAuQDZANsA4QDjAQMBBQEKAQwAAAAAAAACAQAAAAAAAAAZAAAAAAAAAAAAAAAAAAABKQ==Ubytes_Sun, 21 Jul 2019 05:07:52 GMTT1406Uclose�/0bc_NSMutableDictionary�bd3\NSDictionary�/0fg_NSHTTPURLResponse�hi3_NSHTTPURLResponse]NSURLResponse    $ ) 2 7 I L b d � � � � � �3:\_b����������������������(.1:CEGNVakmoqsuwy{}��������������������6Ssw}����������             j              O~        h     (                @                      ��� /7� 2}� /Z� ,� ��� ��� C�� "��  �� /n� $09 Y]^ <<I                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 ���                    		         		
       


	
      		         
   		   

  		  
	


  						         ������������������������������������������������_image/vnd.microsoft.icon�_WebResourceFrameName_WebResourceTextEncodingName_Dhttp://pages.cpsc.ucalgary.ca/~hudsonj/CPSC501F19/notMNISTStarter.pyPO#<html><head></head><body><pre style="word-wrap: break-word; white-space: pre-wrap;">#For Google Collab
#try:
#  # %tensorflow_version only exists in Colab.
#  %tensorflow_version 2.x
#except Exception:
#  pass

import tensorflow as tf
import numpy as np

print("--Get data--")
with np.load("notMNIST.npz", allow_pickle=True) as f:
    x_train, y_train = f['x_train'], f['y_train']
    x_test, y_test = f['x_test'], f['y_test']

print("--Process data--")
print(len(y_train))
x_train, x_test = x_train / 255.0, x_test / 255.0
 
print("--Make model--")
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(10, activation='sigmoid')
])
model.compile(optimizer='sgd', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

print("--Fit model--")
model.fit(x_train, y_train, epochs=1, verbose=2)

print("--Evaluate model--")
model_loss, model_acc = model.evaluate(x_test,  y_test, verbose=2)
print(f"Model Loss:    {model_loss:.2f}")
print(f"Model Accuray: {model_acc*100:.1f}%")
</pre></body></html>Ztext/plainUUTF-8    1 3 < K a s � ��B]h���                           