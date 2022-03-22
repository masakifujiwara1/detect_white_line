# detect_white_line
## 使い方
roscoreの立ち上げ
~~~
roscore
~~~
detect_white_line_nodeの起動
~~~
rosrun detect_white_line detect_white_line_node.py
~~~
カメラの起動
~~~
rosrun usb_cam usb_cam_node
~~~
認識開始させる
~~~
rosservice call /start_detect
~~~
## その他
- service 

    /start_detect : Trigger
- topic 
    
    /white_vel : Twist