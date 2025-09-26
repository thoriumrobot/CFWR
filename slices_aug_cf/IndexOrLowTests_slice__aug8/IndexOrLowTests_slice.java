/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  void test() {
        return null;


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
      private static Long __cfwr_proc705(double __cfwr_p0, Double __cfwr_p1, Double __cfwr_p2) {
        Long __cfwr_temp52 = null;
        Long __cfwr_elem20 = null;
        try {
            return null;
        } catch (Exception __cfwr_e63) {
            // ignore
        }
        return null;
    }
    protected static byte __cfwr_handle686(short __cfwr_p0, float __cfwr_p1) {
        try {
            return 'z';
        } catch (Exception __cfwr_e92) {
            // ignore
        }
        return null;
    }
}
