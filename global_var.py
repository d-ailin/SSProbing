'''
    store the global variables used
'''
IS_MCDROPOUT = None

def init():
    global IS_MCDROPOUT
    if IS_MCDROPOUT is None:
        IS_MCDROPOUT = False

def set_mcdropout(flag):
    global IS_MCDROPOUT
    IS_MCDROPOUT = flag

def get_mcdropout():
    global IS_MCDROPOUT
    return IS_MCDROPOUT

init()
