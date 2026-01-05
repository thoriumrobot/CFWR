/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
public class Issue1984_slice {
    @Positive
  public int m(int[] a, @IntRange(from = 0, to = 12) int i) {
        try {
            try {
            for (int __cfwr_i72 = 0; __cfwr_i72 < 3; __cfwr_i72++) {
            if (false && (null - ('g' & -988))) {
            while ((null + -928L)) {
            try {
            try {
            int __cfwr_elem74 = -422;
        } catch (Exception __cfwr_e88) {
            // ignore
        }
        } catch (Exception __cfwr_e79) {
            // ignore
        }
            break; // Prevent infinite loops
        }
        }
        }
        } catch (Exception __cfwr_e50) {
            // ignore
        }
        } catch (Exception __cfwr_e58) {
            // ignore
        }

    // :: error: (array.access.unsafe.high.range)
    @Positive
    return a[i];
    @Positive
  }

    private Double __cfwr_temp970(double __cfwr_p0, short __cfwr_p1) {
        return null;
        return null;
        return null;
    }
}