/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  void test() {
        float __cfwr_temp55 = -62.40f;


    if (index != -1) {
      array[index] = 1;
    }

    @IndexOrHigh("array") int y = index + 1;
    // :: error: (array.access.unsafe.high)
    array[y] = 1;
    if (y < array.length) {
      array[y] = 1;
    }
    // :: error: (assignment)
    index = array.length;
      boolean __cfwr_func955(char __cfwr_p0, char __cfwr_p1, char __cfwr_p2) {
        try {
            Integer __cfwr_obj28 = null;
        } catch (Exception __cfwr_e83) {
            // ignore
        }
        return false;
    }
    private Character __cfwr_proc492(Object __cfwr_p0, Object __cfwr_p1, Integer __cfwr_p2) {
        try {
            if (false && ((-30.46 - true) / (null / 86.10))) {
            Character __cfwr_obj82 = null;
        }
        } catch (Exception __cfwr_e60) {
            // ignore
        }
        while ((42.47f % 'u')) {
            if (((28L & null) % 'f') || (227 & (null ^ false))) {
            return null;
        }
            break; // Prevent infinite loops
        }
        if ((false << (false >> null)) && ((57.50f - null) / (2.60 << null))) {
            if (true || false) {
            try {
            while ((null % (-164L * null))) {
            return null;
            break; // Prevent infinite loops
        }
        } catch (Exception __cfwr_e94) {
            // ignore
        }
        }
        }
        return null;
    }
}
