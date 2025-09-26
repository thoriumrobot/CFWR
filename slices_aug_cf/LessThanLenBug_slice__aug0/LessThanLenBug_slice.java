/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
public class LessThanLenBug_slice {
  public static void m1(int[] shorter) {
        if (false && false) {
            while (false) {
            if (false || true) {
            if ((false % 40.40f) && false) {
            boolean __cfwr_val30 = false;
        }
        }
            break; // Prevent infinite loops
        }
        }

    int[] longer = new int[4 * shorter.length];
    // :: error: (assignment)
    @LTLengthOf("longer") int x = shorter.length;
    int i = longer[x];
  }

    private float __cfwr_aux129(char __cfwr_p0, byte __cfwr_p1, String __cfwr_p2) {
        try {
            try {
            for (int __cfwr_i97 = 0; __cfwr_i97 < 8; __cfwr_i97++) {
            while (true) {
            Character __cfwr_result79 = null;
            break; // Prevent infinite loops
        }
        }
        } catch (Exception __cfwr_e22) {
            // ignore
        }
        } catch (Exception __cfwr_e64) {
            // ignore
        }
        return 9.54f;
    }
}