bplist00�_WebSubresources_WebMainResource��	
^WebResourceURL_WebResourceResponse_WebResourceData_WebResourceMIMEType_)http://pages.cpsc.ucalgary.ca/favicon.icoObplist00�
X$versionY$archiverT$topX$objects ��_NSKeyedArchiver�	_WebResourceResponse��'-.4567OPQRSTUVWXYZ[\]^_`aeU$null� !"#$%&___nsurlrequest_proto_prop_obj_6___nsurlrequest_proto_prop_obj_3R$2___nsurlrequest_proto_prop_obj_0R$3___nsurlrequest_proto_prop_obj_4V$class___nsurlrequest_proto_prop_obj_1R$4R$0___nsurlrequest_proto_prop_obj_5___nsurlrequest_proto_prop_obj_2R$1���	���� ��()$+,WNS.base[NS.relative� ��_)http://pages.cpsc.ucalgary.ca/favicon.ico�/012Z$classnameX$classesUNSURL�13XNSObject#A���m�W ��89:DNWNS.keysZNS.objects�;<=>?@ABC�	�
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
  						         ������������������������������������������������_image/vnd.microsoft.icon�_WebResourceFrameName_WebResourceTextEncodingName_>http://pages.cpsc.ucalgary.ca/~hudsonj/CPSC501F19/grabimage.pyPO�<html><head></head><body><pre style="word-wrap: break-word; white-space: pre-wrap;">from tkinter import *
import pyscreenshot as ImageGrab

#State of mouse
b1 = "up"
def b1down(event):
    global b1
    b1 = "down"
def b1up(event):
    global b1
    b1 = "up"
def motion(event):
    if b1 == "down":
        event.widget.create_oval(event.x,event.y,event.x,event.y, width=16)

#Main to draw window, capture buttons events, and save image
def main():
    root = Tk()
    root.title("Draw")
    drawing_area = Canvas(root,bg="white",width=28*8,height=28*8)
    drawing_area.pack()
    drawing_area.bind("&lt;Motion&gt;", motion)
    drawing_area.bind("&lt;ButtonPress-1&gt;", b1down)
    drawing_area.bind("&lt;ButtonRelease-1&gt;", b1up)
    button=Button(root,fg="green",text="Save",command=lambda:getter(drawing_area))
    button.pack(side=LEFT)
    button=Button(root,fg="green",text="Clear",command=lambda:delete(drawing_area))
    button.pack(side=RIGHT)
    def delete(widget):
        widget.delete("all")
    def getter(widget):
        x=root.winfo_rootx()+widget.winfo_x()
        y=root.winfo_rooty()+widget.winfo_y()
        x1=x+widget.winfo_width()
        y1=y+widget.winfo_height()
        grabbed = ImageGrab.grab()
        grabbed = grabbed.crop((x,y,x1,y1))
        grabbed = grabbed.resize((28,28))
        grabbed = grabbed.convert(mode="L")
        grabbed.save("image.png")
    root.mainloop()
  
if __name__ == "__main__":
    main()
</pre></body></html>Ztext/plainUUTF-8    1 3 < K a s � ��B]h�����                           �