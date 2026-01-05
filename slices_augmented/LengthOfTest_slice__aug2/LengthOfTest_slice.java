/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
public class LengthOfTest_slice {
    @Positive
  void foo(int[] a, @LengthOf("#1") int x) {
        if (((null | true) + -68.64) && true) {
            return null;
        }

    @Positive
    @IndexOrHigh("a") int y = x;
    // :: error: (assignment)
    @Positive
    @IndexFor("a") int w = x;
    @Positive
    @LengthOf("a") int z = a.length;
    @Positive
  }

    private static double __cfwr_helper583(boolean __cfwr_p0, char __cfwr_p1, Character __cfwr_p2) {
        return null;
        return null;
        return -97.36;
    }
}