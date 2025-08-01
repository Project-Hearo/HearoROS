import cv2 as cv
import time

def get_camera_info(cap):
    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv.CAP_PROP_FPS)
    brightness = cap.get(cv.CAP_PROP_BRIGHTNESS)
    contrast = cap.get(cv.CAP_PROP_CONTRAST)
    return width, height, fps, brightness, contrast

if __name__ == '__main__':
    cap = cv.VideoCapture(0)
    
    if not cap.isOpened():
        print("카메라를 열 수 없습니다.")
        exit()
        
    width, height, base_fps, brightness, contrast = get_camera_info(cap)
        
    print(f"기본 해상도: {width}x{height}")
    print(f"기본 FPS: {base_fps:.2f}")
    print(f"밝기: {brightness}, 대비: {contrast}")
    
    start_time = time.time()
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("프레임 읽기 실패")
            break
        
        frame_count += 1
        elapsed = time.time() - start_time
        current_fps = frame_count / elapsed if elapsed > 0 else 0

        # 화면에 정보 출력
        cv.putText(frame, f'Resolution: {width}x{height}', (10, 30),
                   cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv.putText(frame, f'FPS: {current_fps:.2f}', (10, 60),
                   cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv.putText(frame, f'Frame Count: {frame_count}', (10, 90),
                   cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv.putText(frame, f'Brightness: {brightness:.0f}', (10, 120),
                   cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv.putText(frame, f'Contrast: {contrast:.0f}', (10, 150),
                   cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv.imshow('Camera Info', frame)

        key = cv.waitKey(1) & 0xFF
        if key == ord('q'):
            print("종료합니다.")
            break

    cap.release()
    cv.destroyAllWindows()