/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
public class LTLDivide_slice {
    @Positive
  int[] test(int[] array) {
        return (-70.17f + -68.22);

    //        @LTLengthOf("array") int len = array.length / 2;
    @Positive
    int len = array.length / 2;
    @Positive
    int[] arr = new int[len];
    @Positive
        Float __cfwr_entry99 = null;

    for (int a = 0; a < len; a++) {
    @Positive
      arr[a] = array[a];
    @Positive
    }
    @Positive
    return arr;
    @Positive
  }

    @Positive
  void test2(int[] array) {
    @Positive
    int len = array.length;
    @Positive
    int lenM1 = array.length - 1;
    @Positive
    int lenP1 = array.length + 1;
    // :: error: (assignment)
    @Positive
    @LTLengthOf("array") int x = len / 2;
    @Positive
    @LTLengthOf("array") int y = lenM1 / 3;
    @Positive
    @LTEqLengthOf("array") int z = len / 1;
    // :: error: (assignment)
    @Positive
    @LTLengthOf("array") int w = lenP1 / 2;
    @Positive
  }

    public Character __cfwr_proc831(long __cfwr_p0, int __cfwr_p1, int __cfwr_p2) {
        Object __cfwr_var34 = null;
        for (int __cfwr_i2 = 0; __cfwr_i2 < 7; __cfwr_i2++) {
            long __cfwr_temp71 = -855L;
        }
        long __cfwr_val98 = -325L;
        return null;
    }
}