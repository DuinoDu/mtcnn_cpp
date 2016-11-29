# mtcnn

It a cpp version of [mtcnn](https://github.com/kpzhang93/MTCNN_face_detection_alignment), which is a face detection using cnn.

### Requirement
0. ubuntu (I make it in ubuntu, windows should work fine.)
1. caffe: [https://github.com/BVLC/caffe](https://github.com/BVLC/caffe), [my csdn](http://blog.csdn.net/duinodu/article/details/52760587)
2. opencv: [my csdn](http://blog.csdn.net/duinodu/article/details/51804642)
3. eigen: [http://eigen.tuxfamily.org](https://eigen.tuxfamily.org/index.php?title=Main_Page)
4. libigl: [https://github.com/libigl/libigl](https://github.com/libigl/libigl/)
5. qmake
6. gtest(optional) 

### Compile and run
1. 
    ```
    git clone https://github.com/DuinoDu/mtcnn && cd mtcnn
    qtcreator mtcnn.pro
    ```

2. Edit mtcnn.pro and set correct lib path.

3. Set run line arguments in **Projects** tab, such as */home/duino/project/iactive/mtcnn/mtcnn*. It should contain test images.

4. Build && Run

BTW, you can find python version [here](https://github.com/DuinoDu/mtcnn).
