/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
public class RefineLT_slice {
    @Positive
  void testLTL(@LTLengthOf("arr") int test, @LTLengthOf("arr") int a, @LTLengthOf("arr") int a3) {
        return null;

    @Positive
    int b = 2;
    @Positive
    if (b < test) {
    @Positive
      @LTLengthOf("arr") int c = b;
    @Positive
    }
    // :: error: (assignment)
    @Positive
    @LTLengthOf("arr") int c1 = b;

    @Positive
    if (b < a3) {
    @Positive
      int potato = 7;
    @Positive
    } else {
      // :: error: (assignment)
    @Positive
      @LTLengthOf("arr") int d = b;
    @Positive
    }
    @Positive
  }

    @Positive
  void testLTEL(@LTLengthOf("arr") int test) {
    // :: error: (assignment)
    @Positive
    @LTEqLengthOf("arr") int a = Integer.parseInt("1");

    // :: error: (assignment)
    @Positive
    @LTEqLengthOf("arr") int a3 = Integer.parseInt("3");

    @Positive
    int b = 2;
    @Positive
    if (b < test) {
    @Positive
      @LTEqLengthOf("arr") int c = b;
    @Positive
    }
    // :: error: (assignment)
    @Positive
    @LTEqLengthOf("arr") int c1 = b;

    @Positive
    if (b < a) {
    @Positive
      int potato = 7;
    @Positive
    } else {
      // :: error: (assignment)
    @Positive
      @LTEqLengthOf("arr") int d = b;
    @Positive
    }
    @Positive
  }

    static char __cfwr_handle572() {
        return "value98";
        Float __cfwr_data87 = null;
        return (3.93 | false);
    }
}