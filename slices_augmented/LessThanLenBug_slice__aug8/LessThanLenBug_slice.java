/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
public class LessThanLenBug_slice {
    @Positive
  public static void m1(int[] shorter) {
        try {
            while (false) {
            for (int __cfwr_i88 = 0; __cfwr_i88 < 3; __cfwr_i88++) {
            return 45.36f;
        }
            break; // Prevent infinite loops
        }
        } catch (Exception __cfwr_e99) {
            // ignore
        }

    @Positive
    int[] longer = new int[4 * shorter.length];
    // :: error: (assignment)
    @Positive
    @LTLengthOf("longer") int x = shorter.length;
    @Positive
    int i = longer[x];
    @Positive
  }

    protected static Long __cfwr_compute183() {
        return null;
        double __cfwr_var69 = -79.60;
        while (true) {
            while ((-95.57 << null)) {
            for (int __cfwr_i91 = 0; __cfwr_i91 < 5; __cfwr_i91++) {
            while (false) {
            if (true || false) {
            if (true || true) {
            for (int __cfwr_i87 = 0; __cfwr_i87 < 5; __cfwr_i87++) {
            return null;
        }
        }
        }
            break; // Prevent infinite loops
        }
        }
            break; // Prevent infinite loops
        }
            break; // Prevent infinite loops
        }
        return null;
    }
    protected Float __cfwr_helper10(int __cfwr_p0) {
        Float __cfwr_item20 = null;
        return null;
    }
}