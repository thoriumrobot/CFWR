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
        try {
            Object __cfwr_item41 = null;
        } catch (Exception __cfwr_e43) {
            // ignore

        for (int __cfwr_i19 = 0; __cfwr_i19 < 6; __cfwr_i19++) {
            return null;
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
      protected static int __cfwr_proc44() {
        return null;
        try {
            Boolean __cfwr_var73 = null;
        } catch (Exception __cfwr_e8) {
            // ignore
        }
        try {
            if (true && false) {
            try {
            while (true) {
            if (true || true) {
            while ((null + null)) {
            while (((true << -14.77) + null)) {
            while (true) {
            try {
            try {
            return null;
        } catch (Exception __cfwr_e17) {
            // ignore
        }
        } catch (Exception __cfwr_e62) {
            // ignore
        }
            break; // Prevent infinite loops
        }
            break; // Prevent infinite loops
        }
            break; // Prevent infinite loops
        }
        }
            break; // Prevent infinite loops
        }
        } catch (Exception __cfwr_e51) {
            // ignore
        }
        }
        } catch (Exception __cfwr_e65) {
            // ignore
        }
        return ('0' / false);
    }
}

  void test4(@NonNegative int x1, int[] a) {
