bplist00�_WebSubresources_WebMainResource��	
^WebResourceURL_WebResourceResponse_WebResourceData_WebResourceMIMEType_)http://pages.cpsc.ucalgary.ca/favicon.icoObplist00�
X$versionY$archiverT$topX$objects ��_NSKeyedArchiver�	_WebResourceResponse��'-.4567OPQRSTUVWXYZ[\]^_`aeU$null� !"#$%&___nsurlrequest_proto_prop_obj_6___nsurlrequest_proto_prop_obj_3R$2___nsurlrequest_proto_prop_obj_0R$3___nsurlrequest_proto_prop_obj_4V$class___nsurlrequest_proto_prop_obj_1R$4R$0___nsurlrequest_proto_prop_obj_5___nsurlrequest_proto_prop_obj_2R$1���	���� ��()$+,WNS.base[NS.relative� ��_)http://pages.cpsc.ucalgary.ca/favicon.ico�/012Z$classnameX$classesUNSURL�13XNSObject#A���a� ��89:DNWNS.keysZNS.objects�;<=>?@ABC�	�
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
  						         ������������������������������������������������_image/vnd.microsoft.icon�_WebResourceFrameName_WebResourceTextEncodingName_<http://pages.cpsc.ucalgary.ca/~hudsonj/CPSC501F19/predict.pyPO�<html><head></head><body><pre style="word-wrap: break-word; white-space: pre-wrap;">import sys
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pyscreenshot as ImageGrab

def main():
     class_names = check_args()
     print(f"--Load Model {sys.argv[2]}--")
     #Load the model that should be in sys.argv[2]
     model = None
     print(f"--Load Image {sys.argv[3]}--")
     img = plt.imread(sys.argv[3])
     if np.amax(img.flatten()) &gt; 1:
          img = img/255
     img = 1 - img
     print(f"--Predict as Class {sys.argv[4]}--")
     predict(model, class_names, img, int(sys.argv[4]))

def predict(model, class_names, img, true_label):
    img = np.array([img])
    #Replace these two lines with code to make a prediction
    prediction = [1/10,1/10,1/10,1/10,1/10,1/10,1/10,1/10,1/10,1/10]
    #Determine what the predicted label is
    predicted_label = 0
    plot(class_names, prediction, true_label, predicted_label, img[0])
    plt.show()

def check_args():
     if(len(sys.argv) == 1):
         print("No arguments so using defaults")
         sys.argv = ["predict.py", "MNIST", "MNIST.h5", "image.png", input("Predict the class of image.png(index):")]
     if(len(sys.argv) != 5):
          print("Usage python predict.py &lt;MNIST,notMNIST&gt; &lt;model.h5&gt; &lt;image.png&gt; &lt;prediction class index&gt;")
          sys.exit(1)
     if sys.argv[1] == "MNIST":
          print("--Dataset MNIST--")
          class_names = list(range(10))
     elif sys.argv[1] == "notMNIST":
          print("--Dataset notMNIST--")
          class_names = ["A","B","C","D","E","F","G","H","I","J"]
     else:
          print(f"Choose MNIST or notMNIST, not {sys.argv[1]}")
          sys.exit(2)
     if sys.argv[2][-3:] != ".h5":
          print(f"{sys.argv[2]} is not a h5 extension")
          sys.exit(3)
     if sys.argv[3][-4:] != ".png":
          print(f"{sys.argv[3]} is not a png extension")
          sys.exit(3)
     img = plt.imread(sys.argv[3])
     if len(img.shape) != 2:
          print("Image is not grey scale!")
          sys.exit(4)
     if img.shape != (28,28):
          print("Image is not 28 by 28!")
          sys.exit(4)
     if not sys.argv[4].isdigit():
          print(f"{sys.argv[4]} is not an integer (0-9)")
          sys.exit(3)
     if int(sys.argv[4]) &lt; 0 or int(sys.argv[4]) &gt; 9 :
          print(f"{sys.argv[4]} is not an integer (0-9)")
          sys.exit(3)
     return class_names

def plot(class_names, prediction, true_label, predicted_label, img):
    plt.figure(figsize=(6,3))
    plt.subplot(1,2,1)
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img, cmap=plt.cm.binary)
    predicted_label = np.argmax(prediction)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'
    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],100*np.max(prediction),class_names[true_label]),color=color)
    plt.subplot(1,2,2)
    plt.grid(False)
    plt.xticks(range(10))
    plt.yticks([])
    thisplot = plt.bar(class_names, prediction, color="#777777")
    plt.ylim([0, 1])
    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')

if __name__ == "__main__":
    main()
</pre></body></html>Ztext/plainUUTF-8    1 3 < K a s � ��B]h�����                           �