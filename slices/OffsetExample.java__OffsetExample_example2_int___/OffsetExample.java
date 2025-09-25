import java.util.List;
import org.checkerframework.checker.index.qual.IndexFor;
import org.checkerframework.checker.index.qual.IndexOrHigh;
import org.checkerframework.common.value.qual.MinLen;

public class OffsetExample {

    void example2(int @MinLen(2) [] a) {
        int j = 2;
        int x = a.length;
        int y = x - j;
        a[y] = 0;
        for (int i = 0; i < y; i++) {
            a[i + j] = 1;
            a[j + i] = 1;
            a[i + 0] = 1;
            a[i - 1] = 1;
            a[i + 2 + j] = 1;
        }
    }
}
