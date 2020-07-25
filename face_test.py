
#参考という名のコピペ
#https://qiita.com/FukuharaYohei/items/ec6dce7cc5ea21a51a82

import keras.models
import numpy as np
import cv2
cascade_path = "haarcascade_frontalface_default.xml"


from emothion_recog import emotion_recog


# 使用ファイルと入出力ディレクトリ
image_file = "abe.jpg"
image_path = "./inputs/" + image_file
output_path = "./outputs/" + image_file


#ファイル読み込み
image = cv2.imread(image_path)


#グレースケール変換
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#カスケード分類器の特徴量を取得する
cascade = cv2.CascadeClassifier(cascade_path)

#物体認識（顔認識）の実行
#image – CV_8U 型の行列．ここに格納されている画像中から物体が検出されます
#objects – 矩形を要素とするベクトル．それぞれの矩形は，検出した物体を含みます
#scaleFactor – 各画像スケールにおける縮小量を表します
#minNeighbors – 物体候補となる矩形は，最低でもこの数だけの近傍矩形を含む必要があります
#flags – このパラメータは，新しいカスケードでは利用されません．古いカスケードに対しては，cvHaarDetectObjects 関数の場合と同じ意味を持ちます
#minSize – 物体が取り得る最小サイズ．これよりも小さい物体は無視されます
facerect = cascade.detectMultiScale(image_gray, scaleFactor=1.1, minNeighbors=2, minSize=(30, 30))

color = (255, 255, 255) #白

model = keras.models.load_model('./is_face.h5', compile=False)

# 検出した場合
if len(facerect) > 0:
    #検出した顔を囲む矩形の作成
    for rect in facerect:
        x_data = np.asarray([cv2.resize(image_gray[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2]], (64, 64))]) / 255.0
        x_data = x_data[..., np.newaxis]
        result = model.predict(x_data)
        result = 1 - np.argmax(result, axis=1)
        if result[0]:
            cv2.rectangle(image, tuple(rect[0:2]),tuple(rect[0:2]+rect[2:4]), color, thickness=2)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(image,emotion_recog(x_data),(rect[0],rect[1]+rect[3]+30),font,1,color,2,cv2.LINE_AA)
    #認識結果の保存
    cv2.imwrite(output_path, image)

cv2.imshow('image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
