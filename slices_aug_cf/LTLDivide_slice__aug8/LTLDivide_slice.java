/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  void test2(int[] array) {
        return null;

    int len = array.length;
    int lenM1 = array.length - 1;
    int lenP1 = array.length + 1;
    // :: error: (assignment)
    @LTLengthOf("array") int x = len / 2;
    @LTLengthOf("array") int y = lenM1 / 3;
    @LTEqLengthOf("array") int z = len / 1;
    // :: error: (assignment)
    @LTLengthOf("array") int w = lenP1 / 2;
      protected String __cfwr_compute951(char __cfwr_p0
        short __cfwr_obj60 = null;
, Boolean __cfwr_p1) {
        Float __cfwr_var36 = null;
        if (false || true) {
            while (true) {
            while (true) {
            return null;
            break; // Prevent infinite loops
        }
            break; // Prevent infinite loops
        }
        }
        return "data99";
    }
}
