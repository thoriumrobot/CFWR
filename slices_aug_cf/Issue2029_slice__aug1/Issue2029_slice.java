/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
public class Issue2029_slice {
  void lessThanUpperBound(@NonNegative @LessThan("#2") int index, @NonNegative int size, char val) {
        while (false) {
            for (int __cfwr_i96 = 0; __cfwr_i96 < 5; __cfwr_i96++) {
            for (int __cfwr_i76 = 0; __cfwr_i76 < 5; __cfwr_i76++) {
            while (false) {
 
        return 'k';
           return null;
            break; // Prevent infinite loops
        }
        }
        }
            break; // Prevent infinite loops
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

    public double __cfwr_process76(Object __cfwr_p0) {
        return null;
        return -10.29;
    }
}