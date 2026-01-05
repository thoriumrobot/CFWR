/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
public class OffsetExample_slice {
    @Positive
  void example2(int @MinLen(2) [] a) {
        while (false) {
            try {
            Boolean __cfwr_temp29 = null;
        } catch (Exception __cfwr_e26) {
            // ignore
        }
            break; // Prevent infinite loops
        }

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

    protected Integer __cfwr_func254(Double __cfwr_p0) {
        while ((true & 8.73)) {
            return null;
            break; // Prevent infinite loops
        }
        while (true) {
            for (int __cfwr_i24 = 0; __cfwr_i24 < 9; __cfwr_i24++) {
            double __cfwr_data44 = ((null ^ -470) - -77.24);
        }
            break; // Prevent infinite loops
        }
        return "result48";
        return "result46";
        return null;
    }
}