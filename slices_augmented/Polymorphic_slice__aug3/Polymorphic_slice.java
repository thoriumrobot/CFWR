/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
public class Polymorphic_slice {
    @Positive
  int @PolySameLen [] samelen_identity(int @PolySameLen [] a) {
        float __cfwr_obj22 = 42.62f;

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

    static Float __cfwr_temp918(char __cfwr_p0, double __cfwr_p1) {
        return null;
        if (true && true) {
            if ((461L & null) || false) {
            for (int __cfwr_i74 = 0; __cfwr_i74 < 6; __cfwr_i74++) {
            while (false) {
            try {
            return null;
        } catch (Exception __cfwr_e13) {
            // ignore
        }
            break; // Prevent infinite loops
        }
        }
        }
        }
        for (int __cfwr_i81 = 0; __cfwr_i81 < 7; __cfwr_i81++) {
            if ((null / (null | -92.09)) && true) {
            long __cfwr_var65 = (false & false);
        }
        }
        return null;
        return null;
    }
    public Float __cfwr_proc937(short __cfwr_p0, float __cfwr_p1, Character __cfwr_p2) {
        if (true || false) {
            while (false) {
            return "hello75";
            break; // Prevent infinite loops
        }
        }
        return null;
    }
}