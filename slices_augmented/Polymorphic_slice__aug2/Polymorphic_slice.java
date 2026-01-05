/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
public class Polymorphic_slice {
    @Positive
  int @PolySameLen [] samelen_identity(int @PolySameLen [] a) {
        int __cfwr_temp75 = 973;

    @Positive
    int @SameLen("a") [] x = a;
    @Positive
    return a;
    @Positive
  }

    @Positive
  @PolyUpperBound int ubc_identity(@PolyUpperBound int a) {
    @Positive
    return a;
    @Positive
  }

  // SameLen tests
    @Positive
  void samelen_id(int @SameLen("#2") [] a, int[] a2) {
    @Positive
    int[] banana;
    @Positive
    int @SameLen("a2") [] b = samelen_identity(a);
    // :: error: (assignment)
    @Positive
    int @SameLen("banana") [] c = samelen_identity(b);
    @Positive
  }

    private byte __cfwr_handle144() {
        if (false && false) {
            for (int __cfwr_i20 = 0; __cfwr_i20 < 3; __cfwr_i20++) {
            if (('e' / -92.91) || true) {
            int __cfwr_temp44 = -183;
        }
        }
        }
        if (true && false) {
            return -726L;
        }
        return null;
    }
}