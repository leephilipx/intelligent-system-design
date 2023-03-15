import cv2

SIZE_INFO_H = int(cv2.getTextSize('text', cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0][1] * 1.4)
LABELS = ['Honey', 'Jane', 'Philip', 'Veronica', 'Wai Yeong', 'Unknown']
BBOX_COLOUR = 5 * [(0,255,0)] + [(255,255,255)]


def draw_bbox_simple(frame, bbox_list):

    '''Draw bounding boxes on the frame, given the list of tuples (x1, y1, x2, y2)'''

    for (x1, y1, x2, y2) in bbox_list:
        cv2.rectangle(frame, (x1,y1), (x2,y2), (255,0,0), 2)


def draw_bbox(frame, bbox_list, labels):

    '''Draw bounding boxes on the frame, given the list of tuples (x1, y1, x2, y2)'''

    for label, (x1, y1, x2, y2) in zip(labels, bbox_list):
        cv2.rectangle(frame, (x1,y1), (x2,y2), BBOX_COLOUR[label], 2)
        txt = LABELS[label]
        txt_size = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 1)[0]
        cv2.rectangle(frame, (x1,y1-txt_size[1]-8), (x1+txt_size[0]+4,y1-2), (40,40,40), -1)
        cv2.putText(frame, txt, (x1+2, y1-6), cv2.FONT_HERSHEY_SIMPLEX, 0.65, BBOX_COLOUR[label], 1)


def draw_text(frame, text, loc='NW', index=0):

    '''Draw text on the frame, given the location of the text.'''

    if loc == 'NW':
        cv2.putText(frame, text, (10, 20+index*SIZE_INFO_H), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    else:
        raise ValueError('>> Invalid location for text.')