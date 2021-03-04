import cv2.cv2 as cv2
import numpy as np
import matplotlib.pyplot as plt

cap = cv2.VideoCapture('C:\\Users\\hg\\Desktop\\fall detection\\Fall\\14.avi')  # 開啟影片檔(跌倒)
#cap = cv2.VideoCapture('18.avi')  # 開啟影片檔(未跌倒)
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # 取得畫面寬
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # 取得畫面高
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # 使用 XVID 編碼
area = width * height  # 計算畫面面積
ret, frame = cap.read()
sub = cv2.createBackgroundSubtractorMOG2(history = 300, varThreshold = 20, detectShadows = True) #背景相減
avg = cv2.medianBlur(frame,3)   # 平均畫面模糊處理去噪
avg_float = np.float32(avg)  # 平均畫面轉浮點數
frameCnt = 0  # 影片畫格計數器
frameCntlist = [] # 影片畫格計數器陣列
widthlist = [] #寬度陣列
heightlist = [] #高度陣列
ratiolist = [] #長寬比
falling = False #判斷人是否跌倒
while cap.isOpened():
    ret, frame = cap.read()  # 讀取一幅影格
    frameCnt = frameCnt + 1 
    if not ret:  # 若讀取至影片結尾，則跳出
        break
    blur = cv2.medianBlur(frame,3)  # 模糊處理去噪
    diff = cv2.absdiff(avg, blur)  # 計算目前影格與平均影像的差異值
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)  # 將圖片轉為灰階
    ret, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)  # 篩選出變動程度大於門檻值的區域
    kernel = np.ones((2, 2), np.uint8)  # 使用型態轉換函數去除雜訊
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    cnts, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # 產生等高線
    movecatch = [0, False]
    for c in cnts:
        (x, y, w, h) = cv2.boundingRect(c)  # 計算等高線的外框範圍
        ratio = h/w
        if cv2.contourArea(c) < 2000:  # 忽略太小的區域
            continue
        else:
            movecatch[0] = movecatch[0] + 1
            movecatch[1] = True
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)  # 篩選出的區域畫出方形外框
        print("w = ",w)
        print("h = ",h)
        widthlist.append(w) #寬度資料裝進數列
        heightlist.append(h) #高度資料裝進數列
        ratiolist.append(ratio) #長寬比資料裝進數列
        frameCntlist.append(frameCnt) #計數器陣列

    cv2.imshow('frame', frame)  # 顯示偵測結果影像
    if movecatch[1]:  # 找出該影格抓到的動態區域數
        print('影格=', frameCnt, ', 區塊=', movecatch[0], sep='')
    if cv2.waitKey(1) & 0xFF == ord('q'):  # 按下q鍵終止迴圈
        break
    cv2.accumulateWeighted(blur, avg_float, 0.1)  # 更新平均影像
    avg = cv2.convertScaleAbs(avg_float)
for i in range(len(widthlist)): #判斷是否跌倒
    if i >=1:   #若寬高陣列前後元素的值差距過大，表示跌倒
        if ratiolist[i]-ratiolist[i-1]>=1.5:#若寬高陣列前後元素的值差距過大，表示跌倒
            print("跌倒!")
            falling = True
            break
        if ratiolist[i] <= 0.8: #寬度具一定程度大於高度，表示人呈躺臥狀態
            print("跌倒!")
            falling = True
            break
        else:
            falling = False                 
cap.release()
cv2.destroyAllWindows()

plt.plot(frameCntlist,ratiolist) #畫圖 縱軸為長寬比 橫軸為時間序列
plt.xlabel('time')
plt.ylabel('height/width')
plt.show()

print("==============================================")
print('影片分析完成！')
if falling == True:
    print("有跌倒!")
else:
    print("沒有跌倒!")