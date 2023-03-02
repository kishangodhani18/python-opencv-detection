# =====================================================================
#  Usage: python Text_object-Detection.py --prototxt model\MobileNetSSD_deploy.prototxt --model model\MobileNetSSD_deploy.caffemodel --label person
#         python Text_object-Detection.py --prototxt model\MobileNetSSD_deploy.prototxt --model model\MobileNetSSD_deploy.caffemodel --label person --out output.avi
#         python Text_object-Detection.py --prototxt model\MobileNetSSD_deploy.prototxt
#         --model model\MobileNetSSD_deploy.caffemodel --video test.mp4 --label person --out output.avi
#  Note: Requires opencv 3.4.2 or later
# =====================================================================
from imutils.video import VideoStream
from imutils.video import FPS
from imutils.object_detection import non_max_suppression
import time
import pytesseract
import numpy as np
import argparse
import imutils
import cv2
import dlib
import multiprocessing
from imutils.video import FPS
from utils import update_tracker,create_videowriter,forward_passer

# setting up tesseract path

def box_extractor (scores,geometry,min_confidence):
    num_rows,num_cols = scores.shape[2:4]
    rectangles = []
    confidences = []

    for y in range (num_rows):
        scores_data = scores[0,0,y]
        x_data0 = geometry[0,0,y]
        x_data1 = geometry[0,1,y]
        x_data2 = geometry[0,2,y]
        x_data3 = geometry[0,3,y]
        angles_data = geometry[0,4,y]

        for x in range (num_cols):
            if scores_data[x] < min_confidence:
                continue

            offset_x,offset_y = x * 4.0,y * 4.0

            angle = angles_data[x]
            cos = np.cos (angle)
            sin = np.sin (angle)

            box_h = x_data0[x] + x_data2[x]
            box_w = x_data1[x] + x_data3[x]

            end_x = int (offset_x + (cos * x_data1[x]) + (sin * x_data2[x]))
            end_y = int (offset_y + (cos * x_data2[x]) - (sin * x_data1[x]))
            start_x = int (end_x - box_w)
            start_y = int (end_y - box_h)

            rectangles.append ((start_x,start_y,end_x,end_y))
            confidences.append (scores_data[x])

    return rectangles,confidences


def get_arguments ():
    # First Text Detection
    ap = argparse.ArgumentParser ()
    ap.add_argument ('-vt','--video_text',type = str,
                     help = 'path to optional video file')
    ap.add_argument ('-east','--east',type = str,required = True,
                     help = 'path to EAST text detection model')
    # ap.add_argument ('-c','--min_confidence',type = float,default = 0.5,
    #                  help = 'minimum confidence to process a region')
    ap.add_argument ('-w','--width',type = int,default = 320,
                     help = 'resized image width (multiple of 32)')
    ap.add_argument ('-e','--height',type = int,default = 320,
                     help = 'resized image height (multiple of 32)')
    ap.add_argument ('-p','--padding',type = float,default = 0.0,
                     help = 'padding on each ROI border')

    # Second Object Detection,
    ap.add_argument ('-m','--model',type = str,required = True,
                     help = 'path to detection model')
    ap.add_argument ('-pt','--prototxt',type = str,required = True,
                     help = 'path to prototxt file')
    ap.add_argument ('-c','--min_confidence',type = float,default = 0.2,
                     help = 'minimum confidence for a detection')
    ap.add_argument ('-v','--video_detection',type = str,
                     help = 'path to optional video file')
    ap.add_argument ('-l','--label',type = str,required = True,
                     help = 'item to detect & track')
    ap.add_argument ('-o','--output',type = str,
                     help = 'item to detect & track')
    arguments = vars (ap.parse_args ())

    return arguments


def process_detection (roi):
    # recognizing text
    config = '-l eng --oem 1 --psm 7'
    text = pytesseract.image_to_string (roi[0],config = config)
    print ("Detected Text is: ",text)
    return text,roi[1]


def start_tracker (box,label,frame,in_queue,out_queue):
    """
    Starts a tracker and then continuously pushes its updates to a queue
    :param box: detection box to be tracked
    :param label: label for the detection
    :param frame: frame to place the detector on
    :param in_queue: incoming queue of frames
    :param out_queue: queue to push the labelled trackers on
    """

    # initiate tracker
    tracker = dlib.correlation_tracker ()
    rectangle = dlib.rectangle (box[0],box[1],box[2],box[3])
    tracker.start_track (frame,rectangle)

    while True:
        # get frame
        frame = in_queue.get ()

        if frame is not None:
            # push the tracker after updating it
            out_queue.put ((label,update_tracker (tracker,frame)))


def main (classes,proto,model,video,label_input,output,min_confidence):
    in_queues = []
    out_queues = []

    # pre-load detection model
    print ("[INFO] loading detection model...")
    net = cv2.dnn.readNetFromCaffe (prototxt = proto,caffeModel = model)

    print ('[INFO] Starting video stream...')
    if not video:
        # start web-cam feed
        vs = cv2.VideoCapture (0)

    else:
        # start video stream
        vs = cv2.VideoCapture (video)

    # initializing variables
    writer = None

    fps = FPS ().start ()

    # main loop
    while True:

        grabbed,frame = vs.read ()

        if frame is None:
            break

        # resize the frame & convert to RGB color space (dlib needs RGB)
        frame = imutils.resize (frame,width = 600)
        rgb = cv2.cvtColor (frame,cv2.COLOR_BGR2RGB)

        if output is not None and writer is None:
            # initialize output file writer
            writer = create_videowriter (output,30,(frame.shape[1],frame.shape[0]))

        # if no item in the input queue
        if len (in_queues) == 0:
            height,width = frame.shape[:2]
            # get detections from the model
            detections = forward_passer (net,frame,timing = False)

            # loop through all detections
            for i in np.arange (0,detections.shape[2]):

                confidence = detections[0,0,i,2]

                if confidence > min_confidence:
                    index = int (detections[0,0,i,1])
                    label = classes[index]

                    if label != label_input:
                        continue

                    box = detections[0,0,i,3:7] * np.array ([width,height,width,height])
                    bound_box = box.astype ('int')

                    # initiate multiprocessing queues
                    in_q = multiprocessing.Queue ()
                    out_q = multiprocessing.Queue ()
                    in_queues.append (in_q)
                    out_queues.append (out_q)

                    # initiating daemon process
                    p = multiprocessing.Process (target = start_tracker,
                                                 args = (bound_box,label,rgb,in_q,out_q))
                    p.daemon = True
                    p.start ()

                    # draw rectangles around the tracked detections
                    cv2.rectangle (frame,(bound_box[0],bound_box[1]),(bound_box[2],bound_box[3]),(0,255,0),2)
                    cv2.putText (frame,label,(bound_box[0],bound_box[1] - 15),cv2.FONT_HERSHEY_SIMPLEX,
                                 0.45,(0,255,0),2)

        # if detections already identified
        else:

            # push frames
            for in_q in in_queues:
                in_q.put (rgb)

            # get labelled boxes
            for out_q in out_queues:
                label,label_box = out_q.get ()

                # draw rectangles around the tracked detections
                cv2.rectangle (frame,(label_box[0],label_box[1]),(label_box[2],label_box[3]),
                               (0,255,0),2)
                cv2.putText (frame,label,(label_box[0],label_box[1] - 15),cv2.FONT_HERSHEY_SIMPLEX,
                             0.45,(0,255,0),2)

        if writer is not None:
            writer.write (frame)

        # show result
        cv2.imshow ("Tracking",frame)
        key = cv2.waitKey (1) & 0xFF

        # quit if 'q' is pressed
        if key == ord ('q'):
            break

        fps.update ()

    fps.stop ()
    print (f'[INFO] Elapsed time: {round (fps.elapsed (),2)}')
    print (f'[INFO] approximate FPS: {round (fps.fps (),2)}')

    # release video writer end-point
    if writer is not None:
        writer.release ()

    # release video stream end-point
    cv2.destroyAllWindows ()
    vs.release ()


def web_main (classes,proto,model,video,label_input,output,min_confidence):
    in_queues = []
    out_queues = []

    # pre-load detection model
    print ("[INFO] loading detection model...")
    net = cv2.dnn.readNetFromCaffe (prototxt = proto,caffeModel = model)

    print ('[INFO] Starting video stream...')

    # start video stream
    vs = video

    # main loop
    while True:

        grabbed,frame = vs.read ()

        if frame is None:
            break

        # resize the frame & convert to RGB color space (dlib needs RGB)
        frame = imutils.resize (frame,width = 600)
        rgb = cv2.cvtColor (frame,cv2.COLOR_BGR2RGB)

        # if no item in the input queue
        if len (in_queues) == 0:
            height,width = frame.shape[:2]
            # get detections from the model
            detections = forward_passer (net,frame,timing = False)

            # loop through all detections
            for i in np.arange (0,detections.shape[2]):

                confidence = detections[0,0,i,2]

                if confidence > min_confidence:
                    index = int (detections[0,0,i,1])
                    label = classes[index]

                    if label != label_input:
                        continue

                    box = detections[0,0,i,3:7] * np.array ([width,height,width,height])
                    bound_box = box.astype ('int')

                    # initiate multiprocessing queues
                    in_q = multiprocessing.Queue ()
                    out_q = multiprocessing.Queue ()
                    in_queues.append (in_q)
                    out_queues.append (out_q)

                    # initiating daemon process
                    p = multiprocessing.Process (target = start_tracker,
                                                 args = (bound_box,label,rgb,in_q,out_q))
                    p.daemon = True
                    p.start ()

                    # draw rectangles around the tracked detections
                    cv2.rectangle (frame,(bound_box[0],bound_box[1]),(bound_box[2],bound_box[3]),(0,255,0),2)
                    cv2.putText (frame,label,(bound_box[0],bound_box[1] - 15),cv2.FONT_HERSHEY_SIMPLEX,
                                 0.45,(0,255,0),2)

        # if detections already identified
        else:

            # push frames
            for in_q in in_queues:
                in_q.put (rgb)

            # get labelled boxes
            for out_q in out_queues:
                label,label_box = out_q.get ()

                # draw rectangles around the tracked detections
                cv2.rectangle (frame,(label_box[0],label_box[1]),(label_box[2],label_box[3]),
                               (0,255,0),2)
                cv2.putText (frame,label,(label_box[0],label_box[1] - 15),cv2.FONT_HERSHEY_SIMPLEX,
                             0.45,(0,255,0),2)

        # show result
        ret,jpeg = cv2.imencode ('.jpg',frame)
        frame = jpeg.tobytes ()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


if __name__ == '__main__':

    args = get_arguments ()

    # Text Detection
    # initialize the width & height variables
    w,h = None,None
    new_w,new_h = args['width'],args['height']
    ratio_w,ratio_h = None,None

    # layers which provide a text ROI
    layer_names = ['feature_fusion/Conv_7/Sigmoid','feature_fusion/concat_3']

    # pre-loading the frozen graph
    print ("[INFO] loading EAST text detector...")
    net = cv2.dnn.readNet (args["east"])

    if not args.get ('video',False):
        # start webcam feed
        print ("[INFO] starting video stream...")
        vs = VideoStream (src = 0).start ()
        time.sleep (1)

    else:
        # load video
        vs = cv2.VideoCapture (args['video'])

    fps = FPS ().start ()

    # main loop
    while True:

        # read frame
        frame = vs.read ()
        frame = frame[1] if args.get ('video',False) else frame

        if frame is None:
            break

        # resize frame
        frame = imutils.resize (frame,width = 1000)
        orig = frame.copy ()
        orig_h,orig_w = orig.shape[:2]

        if w is None or h is None:
            h,w = frame.shape[:2]
            ratio_w = w / float (new_w)
            ratio_h = h / float (new_h)

        frame = cv2.resize (frame,(new_w,new_h))

        # getting results from the model
        blob = cv2.dnn.blobFromImage (frame,1.0,(new_w,new_h),(123.68,116.78,103.94),
                                      swapRB = True,crop = False)
        net.setInput (blob)
        scores,geometry = net.forward (layer_names)

        # decoding results from the model
        rectangles,confidences = box_extractor (scores,geometry,min_confidence = args['min_confidence'])
        # applying non-max suppression to get boxes depicting text regions
        boxes = non_max_suppression (np.array (rectangles),probs = confidences)

        # collecting roi from the frame
        roi_list = []
        for (start_x,start_y,end_x,end_y) in boxes:
            start_x = int (start_x * ratio_w)
            start_y = int (start_y * ratio_h)
            end_x = int (end_x * ratio_w)
            end_y = int (end_y * ratio_h)

            dx = int ((end_x - start_x) * args['padding'])
            dy = int ((end_y - start_y) * args['padding'])

            start_x = max (0,start_x - dx)
            start_y = max (0,start_y - dy)
            end_x = min (orig_w,end_x + (dx * 2))
            end_y = min (orig_h,end_y + (dy * 2))

            # ROI to be recognized
            roi = orig[start_y:end_y,start_x:end_x]
            roi_list.append ((roi,(start_x,start_y,end_x,end_y)))

        # recognizing text in roi
        if roi_list:
            # print('creating pool')
            a_pool = multiprocessing.Pool (8)
            # print('starting processes')
            results = a_pool.map (process_detection,roi_list)

            a_pool.close ()
            # a_pool.join()

            # draw results & labels
            for text,box in results:
                start_x,start_y,end_x,end_y = box
                cv2.rectangle (orig,(start_x,start_y),(end_x,end_y),(0,255,0),2)
                cv2.putText (orig,text,(start_x,start_y - 20),
                             cv2.FONT_HERSHEY_SIMPLEX,1.2,(0,0,255),3)

        fps.update ()

        # display result
        cv2.imshow ("Detection",orig)
        key = cv2.waitKey (1) & 0xFF

        # break if 'q' is pressed
        if key == ord ('q'):
            break

    fps.stop ()
    print (f"[INFO] elapsed time {round (fps.elapsed (),2)}")
    print (f"[INFO] approx. FPS : {round (fps.fps (),2)}")

    # Object Detection
    # classes that the model can recognize // change according to the model
    CLASSES = ["background","aeroplane","bicycle","bird","boat",
               "bottle","bus","car","cat","chair","cow","diningtable",
               "dog","horse","motorbike","person","pottedplant","sheep",
               "sofa","train","tvmonitor"]

    main (classes = CLASSES,proto = args['prototxt'],model = args['model'],video = args.get ('video',False),
          label_input = args['label'],output = args.get ('output',None),min_confidence = args['min_confidence'])

    # cleanup
    if not args.get ('video',False):
        vs.stop ()

    else:
        vs.release ()

    cv2.destroyAllWindows ()
