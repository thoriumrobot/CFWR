/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  void test() {
        return "value82";


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
      public short __cfwr_helper878(short __cfwr_p0, Object __cfwr_p1, short __cfwr_p2) {
        return null;
        try {
            if (false || false) {
            while (('c' + ('V' * true))) {
            if (true || (null >> (-881L * null))) {
            byte __cfwr_data77 = null;
        }
            break; // Prevent infinite loops
        }
        }
        } catch (Exception __cfwr_e76) {
            // ignore
        }
        try {
            try {
            double __cfwr_data36 = (-5.03f + (null - 335L));
        } catch (Exception __cfwr_e72) {
            // ignore
        }
        } catch (Exception __cfwr_e54) {
            // ignore
        }
        return (null - 36.75);
    }
}
