def isiterable(p_object):
    try:
        it = iter(p_object)
    except:
        return False
    return True
