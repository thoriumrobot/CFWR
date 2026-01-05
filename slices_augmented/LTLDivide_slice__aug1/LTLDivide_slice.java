/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
public class LTLDivide_slice {
    @Positive
  int[] test(int[] array) {
        for (int __cfwr_i75 = 0; __cfwr_i75 < 8; __cfwr_i75++) {
            Double __cfwr_elem44 = null;
        }

    //        @LTLengthOf("array") int len = array.length / 2;
    @Positive
    int len = array.length / 2;
    @Positive
    int[] arr = new int[len];
    @Positive
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

    protected static char __cfwr_func522(String __cfwr_p0) {
        for (int __cfwr_i69 = 0; __cfwr_i69 < 4; __cfwr_i69++) {
            long __cfwr_obj12 = ((null % false) >> -56.45f);
        }
        return 'y';
    }
}