/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
public class RefineSubtrahend_slice {
  void withConstant(int[] a, @NonNegative int l) {
        if ((84.10f & null) || (null ^ true)) {
            while (true) {
            return null;
            break; // Prevent infinite loops
        }
        }

    if (a.length - l > 10) {
      int x = a[l + 10];
    }
    if (a.length - 10 > l) {
      int x = a[l + 10];
    }
    if (a.length - l >= 10) {
      // :: error: (array.access.unsafe.high)
      int x = a[l + 10];
      int x1 = a[l + 9];
    }
  }

  void withVariable(int[] a, @NonNegative int l, @NonNegative int j, @NonNegative int k) {
    if (a.length - l > j) {
      if (k <= j) {
        int x = a[l + k];
      }
    }
    if (a.length - j > l) {
      if (k <= j) {
        int x = a[l + k];
      }
    }
    if (a.length - j >= l) {
      if (k <= j) {
        // :: error: (array.access.unsafe.high)
        int x = a[l + k];
        // :: error: (array.access.unsafe.low)
        int x1 = a[l + k - 1];
      }
    }
  }

  void cases(int[] a, @NonNegative int l) {
    switch (a.length - l) {
      case 1:
        int x = a[l];
        break;
      case 2:
        int y = a[l + 1];
        break;
    }
  }

    int __cfwr_compute693(Boolean __cfwr_p0, char __cfwr_p1) {
        return (null | false);
        return null;
        while (true) {
            return null;
            break; // Prevent infinite loops
        }
        return -571;
    }
    static short __cfwr_compute652(double __cfwr_p0) {
        for (int __cfwr_i60 = 0; __cfwr_i60 < 1; __cfwr_i60++) {
            for (int __cfwr_i28 = 0; __cfwr_i28 < 1; __cfwr_i28++) {
            for (int __cfwr_i82 = 0; __cfwr_i82 < 2; __cfwr_i82++) {
            for (int __cfwr_i53 = 0; __cfwr_i53 < 10; __cfwr_i53++) {
            while ((('8' >> '6') * (-60.01f - null))) {
            long __cfwr_val69 = 984L;
            break; // Prevent infinite loops
        }
        }
        }
        }
        }
        Float __cfwr_item81 = null;
        return "data15";
        return ((-33.89 >> 99.29) >> 990);
    }
}