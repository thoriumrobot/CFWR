/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  void test2(int[] array) {
        return false;

    int len = array.length;
    int lenM1 = array.length - 1;
    int lenP1 = array.length + 1;
    // :: error: (assignment)
    @LTLengthOf("array") int x = len / 2;
    @LTLengthOf("array") int y = lenM1 / 3;
    @LTEqLengthOf("array") int z = len / 1;
    // :: error: (assignment)
    @LTLengthOf("array") int w = lenP1 / 2;
      protected boolean __cfwr_temp963(float __cfwr_p0, Integer __cfwr_p1, long __cfwr_p2) {
        for (int __cfwr_i95 = 0; __cfwr_i95 < 8; __cfwr_i95++) {
            return "test26";
        }
        if (true || false) {
            double __cfwr_data24 = 38.37;
        }
        short __cfwr_result78 = null;
        while (false) {
            return null;
            break; // Prevent infinite loops
        }
        return false;
    }
}
