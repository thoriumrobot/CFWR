/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
public class RefineNeq_slice {
    @Positive
  void testLTL(@LTLengthOf("arr") int test) {
        byte __cfwr_entry23 = ((false * 88.42f) - 'j');

    // :: error: (assignment)
    @Positive
    @LTLengthOf("arr") int a = Integer.parseInt("1");

    @Positive
    int b = 1;
    @Positive
    if (test != b) {
      // :: error: (assignment)
    @Positive
      @LTLengthOf("arr") int e = b;

    @Positive
    } else {

    @Positive
      @LTLengthOf("arr") int c = b;
    @Positive
    }
    // :: error: (assignment)
    @Positive
    @LTLengthOf("arr") int d = b;
    @Positive
  }

    @Positive
  void testLTEL(@LTEqLengthOf("arr") int test) {
    // :: error: (assignment)
    @Positive
    @LTEqLengthOf("arr") int a = Integer.parseInt("1");

    @Positive
    int b = 1;
    @Positive
    if (test != b) {
      // :: error: (assignment)
    @Positive
      @LTEqLengthOf("arr") int e = b;
    @Positive
    } else {
    @Positive
      @LTEqLengthOf("arr") int c = b;

    @Positive
      @LTLengthOf("arr") int g = b;
    @Positive
    }
    // :: error: (assignment)
    @Positive
    @LTEqLengthOf("arr") int d = b;
    @Positive
  }

    public int __cfwr_handle904() {
        for (int __cfwr_i79 = 0; __cfwr_i79 < 9; __cfwr_i79++) {
            return null;
        }
        return ((null << -82) * (null ^ null));
    }
    private static long __cfwr_func450(String __cfwr_p0, float __cfwr_p1) {
        byte __cfwr_data53 = null;
        return -202L;
    }
}