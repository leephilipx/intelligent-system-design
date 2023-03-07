import cv2


def draw_bbox(frame, bbox_list):

    '''Draw bounding boxes on the frame, given the list of tuples (x, y, w, h)'''

    for (x, y, w, h) in bbox_list:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)

    return frame


def draw_text(frame, text, loc=None):

    '''Draw text on the frame, given the location of the text.'''
    
    if loc == 'NW':
        frame = cv2.putText(frame, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    else:
        raise ValueError('>> Invalid location for text.')

    return frame