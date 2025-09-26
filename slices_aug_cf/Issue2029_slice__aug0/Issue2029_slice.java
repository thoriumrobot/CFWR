/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
public class Issue2029_slice {
  void lessThanUpperBound(@NonNegative @LessThan("#2") int index, @NonNegative int size, char val) {
        try {
            if (true && false) {
            return null;
        }
        } catch (Exception __cfwr_e82) {
            // ignore
        }

    char[] arr = new char[size];
    arr[index] = val;
  }

  void LessThanOffsetLowerBound(
      int[] array, @NonNegative @LTLengthOf("#1") int n, @NonNegative @LessThan("#2 + 1") int k) {
    array[n - k] = 10;
  }

  void LessThanOffsetUpperBound(
      @NonNegative int n,
      @NonNegative @LessThan("#1 + 1") int k,
      @NonNegative int size,
      @NonNegative @LessThan("#3 + 1") int index) {
    @NonNegative int m = n - k;
    int[] arr = new int[size];
    // :: error: (unary.increment)
    for (; index < arr.length - 1; index++) {
      arr[index] = 10;
    }
  }

    char __cfwr_util883(long __cfwr_p0, byte __cfwr_p1) {
        return (-614 * 'k');
        return 'v';
    }
}