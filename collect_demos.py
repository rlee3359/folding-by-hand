from argparse import ArgumentParser

import json
from collections import deque
import copy
import time
import pyrealsense2 as rs
import cv2
import numpy as np
import os
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands


crop_keypoints = np.array([
        [0,0],
        [300, 0],
        [0,300],
        [300, 300],
        ])


rgb_keypoints = np.array([
        [261,80],
        [566,77],
        [260, 380],
        [566, 383]
        ])

# rgb_keypoints = np.array([
#         [165,55],
#         [522,54],
#         [161, 416],
#         [525, 414]
#         ])


ir_keypoints = np.array([
        [298,102],
        [523,100],
        [298, 325],
        [522, 325]
        ])


homography_data = {"crop": crop_keypoints.tolist(), "rgb": rgb_keypoints.tolist(), "ir": ir_keypoints.tolist()}

def crop_mask_gui_img(img):
    rgb_h,_ = cv2.findHomography(rgb_keypoints, crop_keypoints)
    img_warp = cv2.warpPerspective(img, rgb_h, (300,300))

    # Circle mask the image
    # mask = np.zeros((300,300), np.uint8)
    # cv2.circle(mask, (150,150), 150, 255, -1)

    # img_warp = cv2.bitwise_and(img_warp, img_warp, mask=mask)
    return img_warp



def get_cloth_in_frame(ir):
    ir_h,_ = cv2.findHomography(ir_keypoints, crop_keypoints)
    ir_warp = cv2.warpPerspective(ir, ir_h, (300,300))
    cloth_ir_thresh = 30
    mask = 255*np.ones((300,300), np.uint8)
    mask[ir_warp<cloth_ir_thresh] = 0
    num_pix = np.sum(mask)
    # print("Num Pix", num_pix)
    # ir_warp[ir_warp>=cloth_ir_thresh] = 1
    # cv2.imshow("thresh", mask)
    # cv2.waitKey(0)
    
    return num_pix > 500000
 

prev_img = None
def no_movement(img):
    global prev_img
    if prev_img is None:
        prev_img = img
        return False
    
    diff_img = cv2.absdiff(src1=img, src2=prev_img)
    prev_img = img

    # kernel = np.ones((5,5))
    # diff_img = cv2.dilate(diff_img, kernel, 1)

    thresh_img = cv2.threshold(src=diff_img, thresh=100, maxval=255, type=cv2.THRESH_BINARY)[1]
    # cv2.imshow("thresh", thresh_img)
    # cv2.waitKey(1)
    return np.sum(thresh_img) < 17000



parser = ArgumentParser()
parser.add_argument("-t", "--task",  dest="task_name", required=True)
parser.add_argument("-s", "--save",  dest="save_images", default=False)
parser.add_argument("-n", "--num_eps", type=int, dest="num_episodes", default=1)
args = parser.parse_args()


print("Task Selected: " + args.task_name)
print("Writing Data: " + args.save_images)
path = "./data/{}".format(args.task_name)
for i in range(args.num_episodes):
    ep_path = path+"/episode_{}".format(i)
    if not os.path.exists(ep_path): 
        os.makedirs(ep_path)
        print("Created directories.")

with open("{}/homograph_data.json".format(path), 'w', encoding='utf-8') as f:
    json.dump(homography_data, f, ensure_ascii=False, indent=4)


print("Starting program")
# Create a context object. This object owns the handles to all connected realsense devices
pipeline = rs.pipeline()

# Configure streams
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.infrared, 640, 480, rs.format.y8, 30)

# Start streaming
profile = pipeline.start(config)
# align_to = rs.stream.depth
# align = rs.align(align_to)

move_queue = deque()
depth_sensor = profile.get_device().first_depth_sensor()
preset_range = depth_sensor.get_option_range(rs.option.visual_preset)
for i in range(int(preset_range.max)+1):
    visual_preset = depth_sensor.get_option_value_description(rs.option.visual_preset,i)
    print(f"{i}: {visual_preset}")
    if visual_preset == "Short Range":
        print("Found Preset")
        depth_sensor.set_option(rs.option.visual_preset, i)


cv2.imshow("GUI", np.zeros((1000,1000,3)))
cv2.waitKey(0)
circle_crop = True
with mp_hands.Hands(max_num_hands=1, model_complexity=1, min_detection_confidence=0.5, min_tracking_confidence=0.5, static_image_mode=True) as hands:
    # while True:
    for curr_ep in range(args.num_episodes):
        reset_img = np.zeros((480,640,3), np.uint8)
        reset_img = cv2.putText(reset_img, "{}: Reset Cloth".format(curr_ep), (250, 210), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,60,0), 1)
        if circle_crop:
            reset_img = reset_img[100:400, 200:500]

        cv2.imshow("GUI", cv2.resize(reset_img, (1000,1000)))
        cv2.waitKey(0)

        # Initialise Episode
        start = True
        norm = None
        grasped = False
        drop_point = None

        image_capture = False

        within_action = False
        no_hands = True

        ts = 0
        cloth_in_frame = True
        while cloth_in_frame:
            episode_path = path+"/episode_{}".format(curr_ep)
            # This call waits until a new coherent set of frames is available on a device
            # Calls to get_frame_data(...) and get_frame_timestamp(...) on a device will return stable values until wait_for_frames(...) is called
            # print("Waiting for frames")
            frames = pipeline.wait_for_frames()
            # print("Found frames")
            # color = frames.get_color_frame()
            # aligned_frames = align.process(frames)
            depth = frames.get_depth_frame()
            color = frames.get_color_frame()
            ir    = frames.get_infrared_frame()
           
            color = np.asanyarray(color.get_data())

            color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)

            hand_res = hands.process(color)
               
            if not depth or not ir: continue
            # if not depth or not color: continue
            depth = np.asanyarray(depth.get_data())
            ir = np.asanyarray(ir.get_data())

            # color = np.asanyarray(color.get_data())

            color_draw = copy.deepcopy(color)

            hand_res = hands.process(color)
            if hand_res.multi_hand_landmarks:
                no_hands = False
                thumb_tip = hand_res.multi_hand_landmarks[0].landmark[4]
                thumb_tip_xy = np.array([int(thumb_tip.x*640), int(thumb_tip.y*480)])
                index_tip = hand_res.multi_hand_landmarks[0].landmark[8]
                index_tip_xy = np.array([int(index_tip.x*640), int(index_tip.y*480)])

                grasp_dist = np.linalg.norm(index_tip_xy - thumb_tip_xy)
                # If fingers grasp
                grasp_thresh = 30
                if not grasped and grasp_dist < grasp_thresh:
                    print("NEW_GRASP")
                    grasped = True
                    if within_action:
                        print("Broken Grasp")
                    else:
                        within_action = True

                        grasp_point = ((thumb_tip_xy[0] + index_tip_xy[0])//2,
                                       (thumb_tip_xy[1] + index_tip_xy[1])//2)
                    drop_point = None
                elif grasped and grasp_dist >= grasp_thresh:
                    # if within_action:
                    #     print("False Drop")
                    # else:
                    grasped = False
                    drop_point = ((thumb_tip_xy[0] + index_tip_xy[0])//2,
                                  (thumb_tip_xy[1] + index_tip_xy[1])//2)
                if grasped:
                    curr_gripper = ((thumb_tip_xy[0] + index_tip_xy[0])//2,
                                   (thumb_tip_xy[1] + index_tip_xy[1])//2)
                    cv2.circle(color_draw, grasp_point, 20, (0,255,0), 4)
                    cv2.circle(color_draw, curr_gripper, 15, (0,200,50), 4)
                    cv2.arrowedLine(color_draw, grasp_point, curr_gripper, (0,200,0), 2)
                else:
                    pass
                    cv2.circle(color_draw, tuple(thumb_tip_xy), 15, (255,0,0), 2)
                    cv2.circle(color_draw, tuple(index_tip_xy), 15, (255,0,0), 2)
                

                
                # for hand_landmarks in hand_res.multi_hand_landmarks:
                #     mp_drawing.draw_landmarks(
                #         color_draw,
                #         hand_landmarks,
                #         mp_hands.HAND_CONNECTIONS,
                #         mp_drawing_styles.get_default_hand_landmarks_style(),
                #         mp_drawing_styles.get_default_hand_connections_style())
            else:
                no_hands = True

            # print("drop point", drop_point)
            # print("hand landmark", hand_res.multi_hand_landmarks is not None)
            if drop_point is not None:
                cv2.circle(color_draw, drop_point, 20, (0,255,0), 4)
                cv2.arrowedLine(color_draw, grasp_point, drop_point, (0,200,0), 2)



            # depth = depth[36:36+360, 160:160+360]
            # cropped_color = color[36:36+360, 160:160+360]
            # ir = ir[36:36+360, 160:160+360]

            gui_img = crop_mask_gui_img(color_draw)
            cv2.imshow("GUI", cv2.resize(gui_img, (1000,1000)))
            # cv2.imshow("cropped", cropped_color)
            cv2.waitKey(1)

           
            moved = no_movement(color)
            move_queue_len = 30
            if len(move_queue) < 30:
                move_queue.append(moved)
            else:
                move_queue.popleft()
                move_queue.append(moved)

            if np.all(move_queue) and not image_capture and no_hands:
                within_action = False
                image_capture = True
                print("Capture Image {}".format(ts))

                if args.save_images:
                    cv2.imwrite(episode_path + "/{}_rgb.png".format(ts), color)
                    cv2.imwrite(episode_path + "/{}_ir.png".format(ts), ir)
                    np.save(episode_path + "/{}_depth.npy".format(ts), depth)

                    if drop_point is not None:
                        pick = (int(grasp_point[0]), int(grasp_point[1]))
                        place = (int(drop_point[0]), int(drop_point[1]))
                        action_data = {"pick": pick, "place": place}
                        with open("{}/{}_action.json".format(episode_path, ts), 'w', encoding='utf-8') as f:
                            json.dump(action_data, f, ensure_ascii=False, indent=4)
                    flash_time = 50
                    for i in range(flash_time):
                        w = i/flash_time
                        flash_img = (gui_img/255) * w + np.ones_like(gui_img/255) * (1-w)
                        cv2.imshow("GUI", cv2.resize(flash_img, (1000,1000)))
                        cv2.waitKey(1)

                drop_point = None
                grasped = False
                ts+=1
     
            elif not np.all(move_queue):
                image_capture = False

            cloth_in_frame = get_cloth_in_frame(ir)
