#!/usr/bin/env python3

import cv2
import argparse
from background_remover import *

def parse_args():
    parser = argparse.ArgumentParser(description='Real-time background removal')
    parser.add_argument('--camera', type=int, default=0, help='Camera device ID')
    parser.add_argument('--width', type=int, default=640, help='Frame width')
    parser.add_argument('--height', type=int, default=480, help='Frame height')
    parser.add_argument('--mode', type=str, default='green',
                       choices=['green', 'blur', 'image'],
                       help='Background mode: green (solid color), blur, or image')
    parser.add_argument('--bg_image', type=str, default=None,
                       help='Path to background image (for image mode)')
    parser.add_argument('--show_fps', action='store_true',
                       help='Show FPS counter on screen',default=True)

    return parser.parse_args()


def main():
    args = parse_args()

    video_processor = VideoProcessor(
        camera_id=args.camera,
        resolution=(args.width, args.height)
    )

    bg_remover = BackgroundRemover(model_selection=1)

    bg_image = None
    if args.mode == 'image' and args.bg_image:
        bg_image = cv2.imread(args.bg_image)
        if bg_image is None:
            print(f"Error: Could not load background image from {args.bg_image}")
            return

    print("Press 'q' to quit")
    print("Press 'g' for green background")
    print("Press 'b' for blurred background")
    print("Press 'i' for image background (if loaded)")
    print("Press 'f' to toggle FPS display")
    print("Press 'o' to toggle processing ON/OFF")

    current_mode = args.mode
    processing_enabled = True

    try:
        while True:
            frame = video_processor.read_frame()
            if frame is None:
                break

            if processing_enabled:
                if current_mode == 'green':
                    result = bg_remover.remove_background(frame, background_color=(0, 255, 0))
                elif current_mode == 'blur':
                    result = bg_remover.blur_background(frame, blur_strength=61)
                elif current_mode == 'image' and bg_image is not None:
                    result = bg_remover.replace_background_with_image(frame, bg_image)
                else:
                    result = frame
            else:
                result = frame

            fps = video_processor.calculate_fps()

            if args.show_fps:
                cv2.putText(result, f'FPS: {fps:.1f}', (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            cv2.putText(result, f'Mode: {current_mode}', (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            cv2.putText(result, f'Processing: {"ON" if processing_enabled else "OFF"}', (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

            cv2.imshow('Background Removal', result)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('g'):
                current_mode = 'green'
                print("Switched to green background mode")
            elif key == ord('b'):
                current_mode = 'blur'
                print("Switched to blurred background mode")
            elif key == ord('i') and bg_image is not None:
                current_mode = 'image'
                print("Switched to image background mode")
            elif key == ord('f'):
                args.show_fps = not args.show_fps
                print(f"FPS display: {'ON' if args.show_fps else 'OFF'}")
            elif key == ord('o'):
                processing_enabled = not processing_enabled
                print(f"Processing: {'ENABLED' if processing_enabled else 'DISABLED'}")

    except KeyboardInterrupt:
        print("\nInterrupted by user")

    finally:
        video_processor.release()
        cv2.destroyAllWindows()

        avg_fps = video_processor.get_average_fps()
        print(f"\nPerformance summary:")
        print(f"  Resolution: {args.width}x{args.height}")
        print(f"  Average FPS: {avg_fps:.1f}")
        print(f"  Device: CPU")
        print("Done!")


if __name__ == "__main__":
    main()