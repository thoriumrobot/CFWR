/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
public class OffsetExample_slice {
    @Positive
  void example2(int @MinLen(2) [] a) {
        for (int __cfwr_i36 = 0; __cfwr_i36 < 1; __cfwr_i36++) {
            while ((13.70 << 61.67f)) {
            try {
            while (true) {
            try {
            return null;
        } catch (Exception __cfwr_e98) {
            // ignore
        }
            break; // Prevent infinite loops
        }
        } catch (Exception __cfwr_e67) {
            // ignore
        }
            break; // Prevent infinite loops
        }
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

    protected long __cfwr_util911(Long __cfwr_p0) {
        return null;
        while (true) {
            Integer __cfwr_val66 = null;
            break; // Prevent infinite loops
        }
        try {
            while (true) {
            for (int __cfwr_i87 = 0; __cfwr_i87 < 3; __cfwr_i87++) {
            return null;
        }
            break; // Prevent infinite loops
        }
        } catch (Exception __cfwr_e58) {
            // ignore
        }
        return 991L;
    }
}