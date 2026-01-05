/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
public class OffsetExample_slice {
    @Positive
  void example2(int @MinLen(2) [] a) {
        float __cfwr_val4 = 73.66f;

    @Positive
    int j = 2;
    @Positive
    int x = a.length;
    @Positive
    int y = x - j;
    @Positive
    a[y] = 0;
    @Positive
    for (int i = 0; i < y; i++) {
    @Positive
      a[i + j] = 1;
    @Positive
      a[j + i] = 1;
    @Positive
      a[i + 0] = 1;
    @Positive
      a[i - 1] = 1;
      // ::error: (array.access.unsafe.high)
    @Positive
      a[i + 2 + j] = 1;
    @Positive
    }
    @Positive
  }

    @Positive
  void example3(int @MinLen(2) [] a) {
    @Positive
    int j = 2;
    @Positive
    for (int i = 0; i < a.length - 2; i++) {
    @Positive
      a[i + j] = 1;
    @Positive
    }
    @Positive
  }

    public Long __cfwr_calc169(short __cfwr_p0) {
        return null;
        for (int __cfwr_i14 = 0; __cfwr_i14 < 6; __cfwr_i14++) {
            return (-926 / false);
        }
        while (false) {
            return null;
            break; // Prevent infinite loops
        }
        return null;
    }
    public static Object __cfwr_proc100(Double __cfwr_p0, Character __cfwr_p1) {
        try {
            return null;
        } catch (Exception __cfwr_e20) {
            // ignore
        }
        for (int __cfwr_i87 = 0; __cfwr_i87 < 3; __cfwr_i87++) {
            try {
            Object __cfwr_entry29 = null;
        } catch (Exception __cfwr_e28) {
            // ignore
        }
        }
        for (int __cfwr_i82 = 0; __cfwr_i82 < 3; __cfwr_i82++) {
            try {
            short __cfwr_data98 = null;
        } catch (Exception __cfwr_e90) {
            // ignore
        }
        }
        while (false) {
            while (false) {
            while (false) {
            try {
            while ((null + -173L)) {
            try {
            for (int __cfwr_i33 = 0; __cfwr_i33 < 2; __cfwr_i33++) {
            return false;
        }
        } catch (Exception __cfwr_e86) {
            // ignore
        }
            break; // Prevent infinite loops
        }
        } catch (Exception __cfwr_e11) {
            // ignore
        }
            break; // Prevent infinite loops
        }
            break; // Prevent infinite loops
        }
            break; // Prevent infinite loops
        }
        return null;
    }
    static float __cfwr_func234(String __cfwr_p0, int __cfwr_p1, Long __cfwr_p2) {
        double __cfwr_entry98 = -40.15;
        return null;
        float __cfwr_elem37 = ((true ^ 709) - (-76.95f & null));
        return 36.70f;
    }
}