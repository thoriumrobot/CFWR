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
        if (true && (null | 401)) {
            if ((272 & false) || true) {
            if (false || (null + 424L)) {
        return null;

            if ((-651L | null) && true) {
            if (true && true) {
            while (false) {
            return null;
            break; // Prevent infinite loops
        }
        }
        }
        }
        }
        }

    x = x1;
    // :: error: (to.not.ltel)
    b = a;
  }

  void test3(@NonNegative @LessThan("y") int x1, int[] a) {
    x = x1;
    // :: error: (to.not.ltel)
    b = a;
      private static Double __cfwr_compute136() {
        try {
            return null;
        } catch (Exception __cfwr_e18) {
            // ignore
        }
        return null;
    }
}

  void test4(@NonNegative int x1, int[] a) {
