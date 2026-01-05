/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
public class IntroSub_slice {
    @Positive
  void test(int[] arr) {
        Object __cfwr_result95 = null;

    // :: error: (assignment)
    @Positive
    @LTLengthOf({"arr"}) int a = 3;
    @Positive
  }

    @Positive
  void test(int[] arr, @LTLengthOf({"#1"}) int a) {
    // :: error: (assignment)
    @Positive
    @LTLengthOf({"arr"}) int c = a - (-1);
    @Positive
    @LTEqLengthOf({"arr"}) int c1 = a - (-1);
    @Positive
    @LTLengthOf({"arr"}) int d = a - 0;
    @Positive
    @LTLengthOf({"arr"}) int e = a - 7;
    // :: error: (assignment)
    @Positive
    @LTLengthOf({"arr"}) int f = a - (-7);

    // :: error: (assignment)
    @Positive
    @LTEqLengthOf({"arr"}) int j = 7;
    @Positive
  }

    Float __cfwr_calc356(long __cfwr_p0, byte __cfwr_p1, double __cfwr_p2) {
        char __cfwr_item81 = (null / 'R');
        return null;
    }
}