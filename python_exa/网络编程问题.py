# -*- coding: utf-8 -*-

# @Time    : 2019-03-20 16:53
# @Author  : jian
# @File    : 网络编程问题.py

"""
tcp服务器一般都需要绑定，否则客户端找不到服务器
tcp客户端一般不绑定，因为是主动链接服务器，所以只要确定好服务器的ip、port等信息就好，本地客户端可以随机
tcp服务器中通过listen可以将socket创建出来的主动套接字变为被动的，这是做tcp服务器时必须要做的
当客户端需要链接服务器时，就需要使用connect进行链接，udp是不需要链接的而是直接发送，但是tcp必须先链接，只有链接成功才能通信
当一个tcp客户端连接服务器时，服务器端会有1个新的套接字，这个套接字用来标记这个客户端，单独为这个客户端服务
listen后的套接字是被动套接字，用来接收新的客户端的连接请求的，而accept返回的新套接字是标识这个新客户端的
关闭listen后的套接字意味着被动套接字关闭了，会导致新的客户端不能够链接服务器，但是之前已经链接成功的客户端正常通信。
关闭accept返回的套接字意味着这个客户端已经服务完毕
当客户端的套接字调用close后，服务器端会recv解阻塞，并且返回的长度为0，因此服务器可以通过返回数据的长度来区别客户端是否已经下线；同理 当服务器断开tcp连接的时候 客户端同样也会收到0字节数据。

1.简述TCP和UDP的区别以及优缺点? (2018-4-16-lxy)
UDP是面向无连接的通讯协议，UDP数据包括目的端口号和源端口号信息。
优点：UDP速度快、操作简单、要求系统资源较少，由于通讯不需要连接，可以实现广播发送
缺点：UDP传送数据前并不与对方建立连接，对接收到的数据也不发送确认信号，发送端不知道数据是否会正确接收，也不重复发送，不可靠。

TCP是面向连接的通讯协议，通过三次握手建立连接，通讯完成时四次挥手
优点：TCP在数据传递时，有确认、窗口、重传、阻塞等控制机制，能保证数据正确性，较为可靠。
缺点：TCP相对于UDP速度慢一点，要求系统资源较多。


2.简述浏览器通过WSGI请求动态资源的过程? (2018-4-16-lxy)
1.发送http请求动态资源给web服务器
2.web服务器收到请求后通过WSGI调用一个属性给应用程序框架
3.应用程序框架通过引用WSGI调用web服务器的方法，设置返回的状态和头信息。
4.调用后返回，此时web服务器保存了刚刚设置的信息
5.应用程序框架查询数据库，生成动态页面的body的信息
6.把生成的body信息返回给web服务器
7.web服务器吧数据返回给浏览器

3.描述用浏览器访问www.baidu.com的过程(2018-4-16-lxy)
先要解析出baidu.com对应的ip地址
要先使用arp获取默认网关的mac地址
组织数据发送给默认网关(ip还是dns服务器的ip，但是mac地址是默认网关的mac地址)
默认网关拥有转发数据的能力，把数据转发给路由器
路由器根据自己的路由协议，来选择一个合适的较快的路径转发数据给目的网关
目的网关(dns服务器所在的网关)，把数据转发给dns服务器
dns服务器查询解析出baidu.com对应的ip地址，并原路返回请求这个域名的client
得到了baidu.com对应的ip地址之后，会发送tcp的3次握手，进行连接
使用http协议发送请求数据给web服务器
web服务器收到数据请求之后，通过查询自己的服务器得到相应的结果，原路返回给浏览器。
浏览器接收到数据之后通过浏览器自己的渲染功能来显示这个网页。
浏览器关闭tcp连接，即4次挥手结束，完成整个访问过程

4.Post和Get请求的区别? (2018-4-16-lxy)
GET请求，请求的数据会附加在URL之后，以?分割URL和传输数据，多个参数用&连接。
URL的编码格式采用的是ASCII编码，而不是uniclde，即是说所有的非ASCII字符都要编码之后再传输。

POST请求：POST请求会把请求的数据放置在HTTP请求包的包体中。上面的item=bandsaw就是实际的传输数据。
因此，GET请求的数据会暴露在地址栏中，而POST请求则不会。
传输数据的大小：
在HTTP规范中，没有对URL的长度和传输的数据大小进行限制。但是在实际开发过程中，对于GET，特定的浏览器和服务器对URL的长度有限制。因此，在使用GET请求时，传输数据会受到URL长度的限制。
对于POST，由于不是URL传值，理论上是不会受限制的，但是实际上各个服务器会规定对POST提交数据大小进行限制，Apache、IIS都有各自的配置。
安全性：
POST的安全性比GET的高。这里的安全是指真正的安全，而不同于上面GET提到的安全方法中的安全，上面提到的安全仅仅是不修改服务器的数据。比如，在进行登录操作，通过GET请求，用户名和密码都会暴露再URL上，因为登录页面有可能被浏览器缓存以及其他人查看浏览器的历史记录的原因，此时的用户名和密码就很容易被他人拿到了。除此之外，GET请求提交的数据还可能会造成Cross-site request frogery攻击。
效率：GET比POST效率高。

5.cookie 和session 的区别？(2018-4-16-lxy)
1、cookie数据存放在客户的浏览器上，session数据放在服务器上。
2、cookie不是很安全，别人可以分析存放在本地的cookie并进行cookie欺骗考虑到安全应当使用session。
3、session会在一定时间内保存在服务器上。当访问增多，会比较占用服务器的性能考虑到减轻服务器性能方面，应当使用cookie。
4、单个cookie保存的数据不能超过4K，很多浏览器都限制一个站点最多保存20个cookie。
5、建议： 将登陆信息等重要信息存放为SESSION 其他信息如果需要保留，可以放在cookie中

6.HTTP协议状态码有什么用，列出你知道的  HTTP  协议的状态码，然后讲出他们都表示什么意思？(2018-4-16-lxy)
通过状态码告诉客户端服务器的执行状态，以判断下一步该执行什么操作。
常见的状态机器码有：
100-199：表示服务器成功接收部分请求，要求客户端继续提交其余请求才能完成整个处理过程。
200-299：表示服务器成功接收请求并已完成处理过程，常用200（OK请求成功）。

300-399：为完成请求，客户需要进一步细化请求。302（所有请求页面已经临时转移到新的url）。	304、307（使用缓存资源）。

400-499：客户端请求有错误，常用404（服务器无法找到被请求页面），403（服务器拒绝访问，权限不够）。

500-599：服务器端出现错误，常用500（请求未完成，服务器遇到不可预知的情况）。

7.说说HTTP和HTTPS区别？（2018-4-23-lxy）
HTTP协议传输的数据都是未加密的，也就是明文的，因此使用HTTP协议传输隐私信息非常不安全，
为了保证这些隐私数据能加密传输，
于是网景公司设计了SSL（Secure Sockets Layer）协议用于对HTTP协议传输的数据进行加密，从而就诞生了HTTPS。
简单来说，HTTPS协议是由SSL+HTTP协议构建的可进行加密传输、身份认证的网络协议，要比http协议安全。

HTTPS和HTTP的区别主要如下：
1、https协议需要到ca申请证书，一般免费证书较少，因而需要一定费用。
2、http是超文本传输协议，信息是明文传输，https则是具有安全性的ssl加密传输协议。
3、http和https使用的是完全不同的连接方式，用的端口也不一样，前者是80，后者是443。
4、http的连接很简单，是无状态的；HTTPS协议是由SSL+HTTP协议构建的可进行加密传输、身份认证的网络协议，比http协议安全。

8.谈一下HTTP协议以及协议头部中表示数据类型的字段？（2018-4-23-lxy）
HTTP 协议是 Hyper Text Transfer Protocol（超文本传输协议）的缩写，是用于从万维网（WWW:World Wide Web）服务器传输超文本到本地浏览器的传送协议。
HTTP 是一个基于 TCP/IP 通信协议来传递数据（HTML 文件， 图片文件， 查询结果等）。
HTTP 是一个属于应用层的面向对象的协议，由于其简捷、快速的方式，适用于分布式超媒体信息系统。它于 1990 年提出，经过几年的使用与发展，得到不断地完善和扩展。目前在 WWW 中使用的是 HTTP/1.0 的第六版，HTTP/1.1 的规范化工作正在进行之中，而且 HTTP-NG(Next Generation of HTTP)的建议已经提出。
HTTP 协议工作于客户端-服务端架构为上。浏览器作为 HTTP 客户端通过URL 向 HTTP 服务端即 WEB 服务器发送所有请求。Web 服务器根据接收到的请求后，向客户端发送响应信息。
表示数据类型字段： Content-Type

9.HTTP请求方法都有什么？（2018-4-23-lxy）
根据HTTP标准，HTTP请求可以使用多种请求方法。
HTTP1.0定义了三种请求方法： GET， POST 和 HEAD方法。

HTTP1.1新增了五种请求方法：OPTIONS， PUT， DELETE， TRACE 和 CONNECT 方法。
1、	GET	请求指定的页面信息，并返回实体主体。
2、HEAD	类似于get请求，只不过返回的响应中没有具体的内容，用于获取报头
3、POST	向指定资源提交数据进行处理请求（例如提交表单或者上传文件）。数据被包含在请求体中。POST请求可能会导致新的资源的建立和/或已有资源的修改。
4、PUT	从客户端向服务器传送的数据取代指定的文档的内容。
5、DELETE	请求服务器删除指定的页面。
6、CONNECT	HTTP/1.1协议中预留给能够将连接改为管道方式的代理服务器。
7、OPTIONS	允许客户端查看服务器的性能。
8、TRACE	回显服务器收到的请求，主要用于测试或诊断。

10.HTTP常见请求头？（2018-4-23-lxy）
1. Host (主机和端口号) 2.Connection (链接类型) 3.Upgrade-Insecure-Requests (升级为HTTPS请求)
4.User-Agent (浏览器名称)5.Accept (传输文件类型)6. Referer (页面跳转处)7.Accept-Encoding（文件编解码格式）8.Cookie （Cookie）x-requested-with :9.XMLHttpRequest  (是Ajax 异步请求)

11.七层模型？ IP ，TCP/UDP ，HTTP ，RTSP ，FTP 分别在哪层？（2018-4-23-lxy）

IP： 网络层
TCP/UDP： 传输层
HTTP、RTSP、FTP： 应用层协议

12.url的形式？（2018-4-23-lxy）
形式： scheme://host[:port#]/path/…/[?query-string][#anchor]
scheme：协议(例如：http， https， ftp)
host：服务器的IP地址或者域名
port：服务器的端口（如果是走协议默认端口，80 or 443）
path：访问资源的路径
query-string：参数，发送给http服务器的数据
anchor：锚（跳转到网页的指定锚点位置）
http://localhost:4000/file/part01/1.2.html
"""