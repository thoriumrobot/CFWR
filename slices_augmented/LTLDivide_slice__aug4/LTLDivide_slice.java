/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
public class LTLDivide_slice {
    @Positive
  int[] test(int[] array) {
        if (false || false) {
            return -52;
        }

    //        @LTLengthOf("array") int len = array.length / 2;
    @Positive
    int len = array.length / 2;
    @Positive
    int[] arr 
        for (int __cfwr_i30 = 0; __cfwr_i30 < 4; __cfwr_i30++) {
            if (true && true) {
            for (int __cfwr_i36 = 0; __cfwr_i36 < 1; __cfwr_i36++) {
            if (true && true) {
            if (true || ((135L - null) & null)) {
            return -66L;
        }
        }
        }
        }
        }
= new int[len];
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

    protected static Long __cfwr_util125() {
        return (845L >> -75.11);
        if ((null | 593L) && false) {
            return null;
        }
        try {
            String __cfwr_entry94 = "temp36";
        } catch (Exception __cfwr_e48) {
            // ignore
        }
        for (int __cfwr_i11 = 0; __cfwr_i11 < 7; __cfwr_i11++) {
            if (true && (true | false)) {
            if (false || ((-127 << 447L) << ('b' - 's'))) {
            Double __cfwr_var17 = null;
        }
        }
        }
        return null;
    }
}