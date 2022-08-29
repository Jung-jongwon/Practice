import cv2

image = cv2.imread("IU_01.jpg", cv2.IMREAD_COLOR)

# 가중치 파일 경로
cascade_filename = 'model_file/haarcascade_frontalface_alt.xml'

# 모델 불러오기
cascade = cv2.CascadeClassifier(cascade_filename)

def imgDetector(img, cascade):
    
    # 영상 압축
    img = cv2.resize(img, dsize=None, fx=1.0, fy=1.0)
    # 그레이 스케일 변환
    gray = cv2. cvtColor(img, cv2.COLOR_BGR2GRAY)
    # cascade 얼굴 탐지 알고리즘
    results = cascade.detectMultiScale(gray,
                                       scaleFactor=1.5,
                                       minNeighbors=5,
                                       minSize=(20,20)
                                       )
    # 결과값 = 탐지된 객체의 경계상자 list
    for box in results:
        # 좌표 추출
        x,y,w,h = box
        # 경계 상자 그리기
        cv2.rectangle(img, (x,y), (x+w, y+h), (255,255,255), thickness=2)
    
    cv2.imshow("TEST", img)
    cv2.waitKey(10000)
# cv2.destroyAllWindows()    
    
imgDetector(image, cascade)
    
