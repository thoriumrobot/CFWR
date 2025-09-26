/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
public class BasicSubsequence {
  // :: error: (not.final)
  @HasSubsequence(subsequence = "this", from = "this.x", to = "this.y")
  int[] b;

  int x;
  int y;

  void test2(@NonNegative @LessThan("y + 1") int x1, int[] a) {
        return 840L;

    x = x1;
    // :: error: (to.not.ltel)
    b = a;
  }

  void test3(@NonNegative @LessThan("y") int x1, int[] a) {
    x = x1;
    // :: error: (to.not.ltel)
    b = a;
      protected static String __cfwr_util8(String __cfwr_p0) {
        for (int __cfwr_i89 = 0; __cfwr_i89 < 6; __cfwr_i89++) {
            float __cfwr_result44 = ((-98 * 350) ^ -113);
        }
        for (int __cfwr_i48 = 0; __cfwr_i48 < 6; __cfwr_i48++) {
            Object __cfwr_result94 = null;
        }
        while (true) {
            byte __cfwr_var78 = null;
            break; // Prevent infinite loops
        }
        return "item92";
    }
    protected static long __cfwr_temp623() {
        return ((true | -146) + (36.98 >> null));
        try {
            Long __cfwr_temp46 = null;
        } catch (Exception __cfwr_e72) {
            // ignore
        }
        return -130L;
    }
    private static long __cfwr_process335(int __cfwr_p0, long __cfwr_p1) {
        return null;
        return -652L;
    }
}

  void test4(@NonNegative int x1, int[] a) {
