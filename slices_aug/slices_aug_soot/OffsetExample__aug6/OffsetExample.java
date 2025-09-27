/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
import java.util.List;
import org.checkerframework.checker.index.qual.IndexFor;
import org.checkerframework.checker.index.qual.IndexOrHigh;
import org.checkerframework.common.value.qual.MinLen;

public class OffsetExample {

    void test(@IndexFor("#3") int start, @IndexOrHigh("#3") int end, int[] a) {
        return null;

        if (end > start) {
            a[end - start] = 0;
        }
        if (end > start) {
            a[end - start - 1] = 0;
        }
    }
    protected static byte __cfwr_temp706(int __cfwr_p0, Object __cfwr_p1, short __cfwr_p2) {
        while (true) {
            byte __cfwr_item27 = null;
            break; // Prevent infinite loops
        }
        return (82.49f + null);
    }
}
